"""Критерии 5-6: слой памяти - схема, горизонты, конфликты, аудит, удаление, свежесть."""

from datetime import date, timedelta

from src.memory.schema import MemoryKind, MemoryRecord, Source, make_expiry
from src.memory.store import MemoryEngine, _recency

TODAY = date.fromisoformat("2026-09-01")


def _mk_record(age_days):
    observed = TODAY - timedelta(days=age_days)
    return MemoryRecord(
        client_id="c9",
        kind=MemoryKind.FACT,
        attr="city",
        value="Moscow",
        observed_at=observed.isoformat(),
        expires_at="2027-09-01",
    )


def test_horizons_differ_per_kind(offline_env):
    from src.memory.schema import RETENTION_DAYS
    exp_fact = make_expiry(MemoryKind.FACT, TODAY)
    assert exp_fact == (TODAY + timedelta(days=RETENTION_DAYS[MemoryKind.FACT])).isoformat()
    assert make_expiry(MemoryKind.EPISODE, TODAY) != exp_fact


def test_conflict_rule_trust_hierarchy(offline_env):
    _, _, engine = offline_env
    engine.write("c1", MemoryKind.FACT, "tariff", "Start", source=Source.USER, actor="t")
    engine.write("c1", MemoryKind.FACT, "tariff", "Enterprise", source=Source.USER_INFERRED, actor="t")
    assert engine.get("c1", MemoryKind.FACT, "tariff").value == "Start"
    engine.write("c1", MemoryKind.FACT, "tariff", "Enterprise", source=Source.VERIFIED, actor="t")
    assert engine.get("c1", MemoryKind.FACT, "tariff").value == "Enterprise"
    ops = [e.op for e in engine.audit_report("c1")]
    assert "REJECT" in ops
    assert "UPDATE" in ops


def test_delete_keeps_mandatory_masked(offline_env):
    _, _, engine = offline_env
    engine.write("c2", MemoryKind.PRODUCT, "payments", "valid-to-dec",
                 source=Source.VERIFIED, mandatory=True, actor="t")
    engine.write("c2", MemoryKind.PREFERENCE, "comm_email", True, source=Source.USER, actor="t")
    report = engine.delete("c2", reason="запрос пользователя")
    purged = [x.lower() for x in report["purged"]]
    kept = [x.lower() for x in report["kept_mandatory"]]
    assert "preference/comm_email" in purged
    assert "product/payments" in kept
    rec = engine.get("c2", MemoryKind.PRODUCT, "payments")
    assert rec.masked
    assert "<masked" in str(rec.value)
    ops = [e.op for e in engine.audit_report("c2")]
    assert "PURGE" in ops
    assert "MASK" in ops


def test_recency_monotonically_decays(offline_env):
    fresh_score = _recency(_mk_record(2), TODAY)
    stale_score = _recency(_mk_record(664), TODAY)
    assert stale_score < fresh_score


def test_search_membership(offline_env):
    _, _, engine = offline_env
    engine.write("c7", MemoryKind.FACT, "home", "Paris")
    engine.write("c7", MemoryKind.PREFERENCE, "language", "english")
    res = engine.search("c7", "город проживания", top_k=4)
    attrs = {(r.kind.value, r.attr) for r, _ in res}
    assert ("fact", "home") in attrs
    assert ("preference", "language") in attrs


def test_snapshot_roundtrip_across_process_like_reload(tmp_path):
    path = str(tmp_path / "mem.json")
    e1 = MemoryEngine(embedder=None, clock=lambda: TODAY, snapshot_path=path)
    e1.write("cx", MemoryKind.FACT, "tariff", "Start", source=Source.USER)
    e1.write("cx", MemoryKind.PREFERENCE, "comm_email", True)
    assert path_exists(path)

    e2 = MemoryEngine(embedder=None, clock=lambda: TODAY, snapshot_path=path)
    assert e2.get("cx", MemoryKind.FACT, "tariff").value == "Start"
    assert e2.get("cx", MemoryKind.PREFERENCE, "comm_email").value is True
    assert len(e2.audit_report("cx")) == 2


def path_exists(fp):
    import os
    return os.path.exists(fp)
