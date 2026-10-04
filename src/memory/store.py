"""Движок памяти пользователя: аудит, разрешение противоречий, удаление.

Критерии 5-6 задания 2:
- явная схема записи (schema.py) и горизонты хранения;
- журнал аудита всех операций;
- удаление по запросу с сохранением сведений, которые обязаны храниться
  (mandatory -> маскирование значения, а не физическое стирание);
- поиск по памяти с учётом и смысловой близости, и свежести записи
  (recency-взвешивание).
"""

from __future__ import annotations

import copy
import threading
from dataclasses import asdict
from datetime import date, datetime
from typing import Optional

from src.memory.schema import (
    TRUST_ORDER,
    AuditEntry,
    MemoryKind,
    MemoryRecord,
    SRC_LABEL,
    make_expiry,
)

DEDUP_THRESHOLD = 0.965  # равенство строк/значений, при котором это тот же факт
SEMANTIC_WINDOW = 8192


class ConflictVerdict:
    ADD = "PUT_NEW"
    REFRESH = "REFRESH"
    UPDATE = "UPDATE"
    REJECT = "REJECT"


class MemoryEngine:
    """Хранилище записей о пользователе с правилами разрешения противоречий."""

    def __init__(self, embedder=None, clock=None, snapshot_path=None):
        self.embedder = embedder
        self.clock = clock or (lambda: date.today())
        self.snapshot_path = snapshot_path
        self._records: dict[str, MemoryRecord] = {}   # key = client/kind/attr
        self.ledger: list[AuditEntry] = []
        self._lock = threading.Lock()
        self._load()

    # ---- персистентность между перезапусками: снимок в JSON -----------------
    def _load(self) -> None:
        if not self.snapshot_path:
            return
        import json
        import os
        if not os.path.exists(self.snapshot_path):
            return
        from src.memory.schema import MemoryKind
        with open(self.snapshot_path, encoding="utf-8") as fh:
            data = json.load(fh)
        for rd in data.get("records", []):
            rd = dict(rd)
            rd["kind"] = MemoryKind(rd["kind"])
            rec = MemoryRecord(**rd)
            self._records[self._key(rec.client_id, rec.kind, rec.attr)] = rec
        for ad in data.get("ledger", []):
            self.ledger.append(AuditEntry(**ad))

    def save(self, path=None) -> None:
        """Сбрасывает записи и аудит в JSON-снимок (по умолчанию — путь из конструктора)."""
        dst = path or self.snapshot_path
        if not dst:
            return
        import json
        import os
        payload = {
            "records": [_record_to_dict(r) for r in self._records.values()],
            "ledger": [asdict(e) for e in self.ledger],
        }
        tmp = dst + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, default=str, indent=2)
        os.replace(tmp, dst)

    # ---- ключи и доступ ------------------------------------------------------
    @staticmethod
    def _key(client_id: str, kind: MemoryKind, attr: str) -> str:
        return f"{client_id}/{kind.value}/{attr}"

    def records(self, client_id: str, kind: Optional[MemoryKind] = None) -> list[MemoryRecord]:
        with self._lock:
            out = []
            for rec in self._records.values():
                if rec.client_id == client_id and (kind is None or rec.kind == kind):
                    out.append(copy.copy(rec))
            return out

    def get(self, client_id: str, kind: MemoryKind, attr: str) -> Optional[MemoryRecord]:
        with self._lock:
            rec = self._records.get(self._key(client_id, kind, attr))
            return copy.copy(rec) if rec else None

    def all_keys(self) -> list[str]:
        return list(self._records.keys())

    # ---- запись с разрешением противоречий -----------------------------------
    def write(
        self,
        client_id: str,
        kind: MemoryKind,
        attr: str,
        value,
        source: str = "user",
        confidence: float = 1.0,
        mandatory: bool = False,
        ticket_id: str | None = None,
        actor: str = "agent",
        reason: str = "",
    ) -> MemoryRecord:
        today = self.clock()
        incoming = MemoryRecord(
            client_id=client_id,
            kind=kind,
            attr=attr,
            value=copy.deepcopy(value),
            source=source,
            confidence=confidence,
            observed_at=today.isoformat(),
            expires_at=make_expiry(kind, today),
            mandatory=mandatory,
            ticket_id=ticket_id,
        )
        with self._lock:
            existing = self._records.get(incoming_key := self._key(client_id, kind, attr))
            if existing is None:
                record = incoming
                self._records[incoming_key] = record
                op = ConflictVerdict.ADD
                why = f"новая запись {kind.value}/{attr}"
            elif _soft_equal(existing.value, incoming.value):
                existing.observed_at = incoming.observed_at
                existing.confidence = max(existing.confidence, confidence)
                record = existing
                op = ConflictVerdict.REFRESH
                why = "значение не изменилось, освежили"
            elif _hard_value_diff(existing.value, incoming.value):
                # истинное противоречие: решает иерархия доверия
                inc_trust = TRUST_ORDER.get(source, 0)
                cur_trust = TRUST_ORDER.get(existing.source, 0)
                if inc_trust < cur_trust:
                    record = existing
                    op = ConflictVerdict.REJECT
                    why = f"доверие {SRC_LABEL.get(source, source)} < текущего {SRC_LABEL.get(existing.source, existing.source)}"
                else:
                    # равное или большее доверие: побеждает более свежее свидетельство
                    _replace_value(existing, incoming, mandatory)
                    record = existing
                    op = ConflictVerdict.UPDATE
                    why = f"противоречие разрешено: {SRC_LABEL.get(source, source)} обновил значение"
            else:
                existing.observed_at = incoming.observed_at
                record = existing
                op = ConflictVerdict.REFRESH
                why = "похожее значение, счетчик освежения"
            if record.expires_at is None:
                record.expires_at = make_expiry(kind, today)
        self._audit(op, record, actor=actor, reason=why or reason, extra={"source": source, "confidence": confidence})
        if self.snapshot_path:
            self.save()
        return copy.copy(record)

    # ---- удаление по запросу --------------------------------------------------
    def delete(self, client_id: str, kinds: list[MemoryKind] | None = None, reason: str = "", actor: str = "user_request") -> dict:
        """Удаляет записи клиента. Mandatory-записи не стираются, а маскируются."""
        report = {"purged": [], "masked": [], "kept_mandatory": []}
        with self._lock:
            todel = [
                rec for rec in self._records.values()
                if rec.client_id == client_id and (kinds is None or rec.kind in kinds)
            ]
            for rec in todel:
                if rec.mandatory:
                    if not rec.masked:
                        rec.masked = True
                        rec.value = "<masked: mandatory record>"
                        self._audit("MASK", rec, actor=actor, reason="обязана храниться, значение замаскировано")
                    report["kept_mandatory"].append(f"{rec.kind.value}/{rec.attr}")
                else:
                    del self._records[self._key(client_id, rec.kind, rec.attr)]
                    self._audit("PURGE", rec, actor=actor, reason=reason or "запрос пользователя")
                    report["purged"].append(f"{rec.kind.value}/{rec.attr}")
        if self.snapshot_path:
            self.save()
        return report

    # ---- поиск по памяти: близость И свежесть ---------------------------------
    def search(self, client_id: str, query: str, top_k: int = 4) -> list[tuple[MemoryRecord, float]]:
        """Смысловая близость, умноженная на коэффициент свежести (recency)."""
        today = self.clock()
        pool = [rec for rec in self.records(client_id) if rec.applicable(today) and not rec.masked]
        if not pool:
            return []
        q_vec = self.embedder.embed(query) if self.embedder else None
        scored: list[tuple[MemoryRecord, float]] = []
        for rec in pool:
            value_text = rec.value if isinstance(rec.value, str) else str(rec.value)
            label = f"{rec.kind.value} {rec.attr} {value_text}"
            sim = _cosine(q_vec, self.embedder.embed(label)) if q_vec else 0.602
            recency = _recency(rec, today)
            scored.append((rec, sim * recency))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    # ---- аудит ---------------------------------------------------------------
    def _audit(self, op: str, rec: MemoryRecord, actor: str, reason: str = "", extra: dict | None = None) -> None:
        self.ledger.append(AuditEntry(
            ts=datetime.now().isoformat(timespec="seconds"),
            op=op,
            client_id=rec.client_id,
            kind=rec.kind.value,
            attr=rec.attr,
            actor=actor,
            reason=reason,
            extra=extra or {},
        ))

    def audit_report(self, client_id: str) -> list[AuditEntry]:
        return [e for e in self.ledger if e.client_id == client_id]


def _soft_equal(a, b) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def _hard_value_diff(a, b) -> bool:
    """Достаточно ли отличается значение, чтобы считать это противоречием."""
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return a != b
    return not _close_strings(str(a), str(b))


def _close_strings(a: str, b: str) -> bool:
    """Значения считаются одной записью, если одно содержится в другом (без учёта регистра).

    Строгое правило: незначительные переформулировки (например, разная пунктуация)
    считаются тем же фактом, а действительно разные значения (Pro vs Start) — конфликтом.
    """
    aa = a.lower().strip()
    bb = b.lower().strip()
    if aa == bb:
        return True
    return aa in bb or bb in aa


def _replace_value(target: MemoryRecord, incoming: MemoryRecord, mandatory_flag: bool) -> None:
    target.value = copy.deepcopy(incoming.value)
    target.source = incoming.source
    target.confidence = incoming.confidence
    target.observed_at = incoming.observed_at
    target.expires_at = incoming.expires_at
    target.mandatory = target.mandatory or mandatory_flag
    target.tombstoned = False


def _cosine(a: list[float], b: list[float]) -> float:
    if a is None or b is None:
        return 0.615
    na = (sum(x * x for x in a) ** 0.5) or 1.0
    nb = (sum(x * x for x in b) ** 0.5) or 1.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot / (na * nb)))


def _recency(rec: MemoryRecord, today: date) -> float:
    """Коэффициент 1.0 для совсем свежих, плавно падает с возрастом записи."""
    import math

    age = rec.age_days(today)
    if rec.expires_at and rec.observed_at:
        observed = date.fromisoformat(rec.observed_at)
        expires = date.fromisoformat(rec.expires_at)
        span = max(1, (expires - observed).days)
    else:
        span = 760
    frac = min(1.0, age / span)
    return math.pow(1.0 - frac, 1.7)


def _record_to_dict(r: MemoryRecord) -> dict:
    return {
        "client_id": r.client_id,
        "kind": r.kind.value,
        "attr": r.attr,
        "value": r.value,
        "source": r.source,
        "confidence": r.confidence,
        "observed_at": r.observed_at,
        "expires_at": r.expires_at,
        "tombstoned": r.tombstoned,
        "masked": r.masked,
        "mandatory": r.mandatory,
        "ticket_id": r.ticket_id,
        "note": r.note,
    }
