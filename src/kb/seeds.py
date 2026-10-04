"""Встроенное наполнение базы знаний: документы и их редакции."""

from __future__ import annotations

from src.kb.catalog import Edition, KnowledgeDoc, register_seed

register_seed(KnowledgeDoc(
    doc_id="doc-start-limits",
    doc_num="06-Т",
    family_title="Тариф Start: лимит параллельных воркеров",
    editions=(
        Edition(
            version=1,
            valid_from="2025-09-01",
            valid_to="2026-01-14",
            title="Тариф Start, ред. 1",
            section="тарифы",
            text=(
                "Тариф Start позволяет одновременно исполнять до 5 параллельных воркеров. "
                "При превышении пайплайн останавливается с ошибкой QUOTA_EXCEEDED. "
                "Для большинства небольших проектов пяти воркеров достаточно. "
                "Повысить временный буфер можно увеличением тарифа."
            ),
        ),
        Edition(
            version=2,
            valid_from="2026-01-15",
            valid_to=None,
            title="Тариф Start, ред. 2",
            section="тарифы",
            text=(
                "С 15 января 2026 года тариф Start позволяет одновременно исполнять до 10 "
                "параллельных воркеров. Ошибка QUOTA_EXCEEDED теперь возможна только при "
                "кратковременном всплеске выше десяти одновременных задач. Существующие "
                "клиенты получили новый лимит автоматически."
            ),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-pro-limits",
    doc_num="16-Т",
    family_title="Тариф Pro: лимит параллельных воркеров",
    editions=(
        Edition(
            version=1,
            valid_from="2025-09-01",
            valid_to="2026-01-31",
            title="Тариф Pro, ред. 1",
            section="тарифы",
            text=(
                "Тариф Pro позволяет одновременно исполнять до 20 параллельных воркеров "
                "и поддерживает настраиваемые правила автоскейлинга. Ошибка QUOTA_EXCEEDED "
                "возникает только при нештатном всплеске."
            ),
        ),
        Edition(
            version=2,
            valid_from="2026-02-01",
            valid_to=None,
            title="Тариф Pro, ред. 2",
            section="тарифы",
            text=(
                "С февраля 2026 года тариф Pro позволяет одновременно исполнять до 50 "
                "параллельных воркеров, получает приоритетную очередь и увеличенный лимит "
                "электронной почты уведомлений."
            ),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-ent-limits",
    doc_num="22-Т",
    family_title="Тариф Enterprise: ресурсы и SLA",
    editions=(
        Edition(
            version=1,
            valid_from="2026-01-01",
            valid_to=None,
            title="Тариф Enterprise, ред. 1",
            section="тарифы",
            text=(
                "Тариф Enterprise предоставляет до 800 параллельных воркеров, приватные "
                "кластеры, выделенные IP-адреса и соглашение об уровне обслуживания 99.9%."
            ),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-quota-rules",
    doc_num="07-Т",
    family_title="Превышение квоты: причины и меры",
    editions=(
        Edition(
            version=1,
            valid_from="2025-09-01",
            valid_to=None,
            title="Диагностика QUOTA_EXCEEDED, ред. 1",
            section="эксплуатация",
            text=(
                "Ошибка QUOTA_EXCEEDED появляется, когда число одновременных задач превышает "
                "лимит текущего тарифа. Порядок действий: проверить количество параллельных "
                "воркеров в проекте, снизить burst-параметр расписания, при регулярных падениях "
                "перейти на тариф с большим лимитом либо включить автоскейлинг. Само по себе "
                "падение воркера данные не теряет: незавершённые шаги становятся в очередь."
            ),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-autoscaling",
    doc_num="12-Р",
    family_title="Автоскейлинг: когда менять настройки",
    editions=(
        Edition(
            version=1,
            valid_from="2025-10-01",
            valid_to=None,
            title="Рекомендации по автоскейлингу, ред. 1",
            section="эксплуатация",
            text=(
                "Настройки автоскейлинга рекомендуется не менять перед плановыми выпусками "
                "и в периоды отпусков, если трафик не изменяется. Правило масштабирования "
                "можно править в любое время, но эффект применяется с ближайшего окна "
                "перерасчёта раз в 5 минут."
            ),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-notifications",
    doc_num="09-Р",
    family_title="Уведомления: email и push",
    editions=(
        Edition(
            version=1,
            valid_from="2025-09-01",
            valid_to="2026-02-28",
            title="SMTP relay, ред. 1",
            section="уведомления",
            text=(
                "Email-уведомления отправляются через релей smtp-legacy.devcloud.io. "
                "Письма консолидируются и уходят раз в час; задержка до часа является нормой. "
                "Push-уведомления в мобильном приложении включены по умолчанию."
            ),
        ),
        Edition(
            version=2,
            valid_from="2026-03-01",
            valid_to=None,
            title="SMTP relay, ред. 2",
            section="уведомления",
            text=(
                "С 1 марта 2026 года email-уведомления отправляются через новый релей "
                "smtp-news.devcloud.io и доставляются мгновенно. Перед переходом необходимо "
                "прописать SPF-запись для домена, иначе письма могут попадать в спам. "
                "Push-уведомления настраиваются в профиле приложения."
            ),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-reports",
    doc_num="14-Р",
    family_title="Экспорт отчётов: расписание и SLA",
    editions=(
        Edition(
            version=1,
            valid_from="2025-11-01",
            valid_to=None,
            title="Расписание экспорта, ред. 1",
            section="эксплуатация",
            text=(
                "Ночная выгрузка отчётов запускается в 02:00 UTC ежедневно. Гарантированная "
                "доставка отчёта в архив до 06:00 UTC. Если ночная выгрузка прервалась, "
                "повторный запуск происходит до 12:00 UTC того же дня, потери данных нет."
            ),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-refund",
    doc_num="03-П",
    family_title="Возврат и перенос оплаты",
    editions=(
        Edition(
            version=1,
            valid_from="2025-09-01",
            valid_to="2026-03-31",
            title="Правила возврата, ред. 1",
            section="биллинг",
            text=(
                "Возврат средств за неиспользованный месяц тарифа возможен, если заявка подана "
                "в течение 30 дней с момента оплаты. Перенос оплаты на следующий период не "
                "предусмотрен."
            ),
        ),
        Edition(
            version=2,
            valid_from="2026-04-01",
            valid_to=None,
            title="Правила возврата, ред. 2",
            section="биллинг",
            text=(
                "С апреля 2026 года возврат средств за неиспользованный месяц возможен при подаче "
                "заявки в течение 14 дней с момента оплаты. Бесплатный перенос оплаты на следующий "
                "период разрешён один раз за год."
            ),
        ),
    ),
))

# ---- расширенная БЗ: сводные правила, API, инциденты, окна, безопасность, биллинг ----

register_seed(KnowledgeDoc(
    doc_id="doc-work-table",
    doc_num="11-Т",
    family_title="Лимиты параллельных воркеров по тарифам (сводная таблица)",
    editions=(
        Edition(
            version=1, valid_from="2025-09-01", valid_to="2026-01-31",
            title="Сводная таблица лимитов, ред. 1", section="тарифы",
            text=("Сводная таблица до 01.02.2026: Start — до 5 параллельных воркеров, "
                  "Pro — до 20, Enterprise — до 800."),
        ),
        Edition(
            version=2, valid_from="2026-02-01", valid_to=None,
            title="Сводная таблица лимитов, ред. 2", section="тарифы",
            text=("Сводная таблица с февраля 2026: Start — до 10 параллельных воркеров, "
                  "Pro — до 50, Enterprise — до 800, выделенный VIP-пул — до 1200."),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-api-rates",
    doc_num="02-Т",
    family_title="Rate limits REST API по тарифам",
    editions=(
        Edition(
            version=1, valid_from="2025-10-01", valid_to="2026-03-31",
            title="Лимиты запросов, ред. 1", section="интеграции",
            text=("Rate limits до 01.04.2026: Start — 120 запросов/мин, "
                  "Pro — 600 запросов/мин, Enterprise — 5000 запросов/мин."),
        ),
        Edition(
            version=2, valid_from="2026-04-01", valid_to=None,
            title="Лимиты запросов, ред. 2", section="интеграции",
            text=("С апреля 2026 лимиты увеличены: Start — 180, Pro — 900, "
                  "Enterprise — 8000 запросов/мин; для Pro добавлен burst x2."),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-incident-slo",
    doc_num="01-РЕ",
    family_title="Реагирование на инциденты и эскалация",
    editions=(
        Edition(
            version=1, valid_from="2025-09-01", valid_to="2026-02-28",
            title="SLA реагирования, ред. 1", section="эксплуатация",
            text=("Время реакции на инциденты до 01.03.2026: стандартный — 30 минут, "
                  "критический — 15 минут."),
        ),
        Edition(
            version=2, valid_from="2026-03-01", valid_to=None,
            title="SLA реагирования, ред. 2", section="эксплуатация",
            text=("С марта 2026 реакция ужесточена: стандартный инцидент — 15 минут, "
                  "критический — 5 минут; критические инциденты эскалируются на L2 сразу."),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-maintenance-window",
    doc_num="17-О",
    family_title="Плановые окна обслуживания",
    editions=(
        Edition(
            version=1, valid_from="2025-11-01", valid_to="2026-04-30",
            title="Окна работ, ред. 1", section="эксплуатация",
            text=("Плановые работы до 01.05.2026: среда 01:00-03:00 UTC, "
                  "предупреждение за 3 рабочих дня."),
        ),
        Edition(
            version=2, valid_from="2026-05-01", valid_to=None,
            title="Окна работ, ред. 2", section="эксплуатация",
            text=("С мая 2026 окна работ: вторник и четверг 00:30-02:30 UTC, "
                  "предупреждение за 7 рабочих дней; учитываются часовые пояса клиентов."),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-security-roles",
    doc_num="25-Р",
    family_title="Роли доступа и безопасность",
    editions=(
        Edition(
            version=1, valid_from="2026-01-01", valid_to=None,
            title="Роли и MFA, ред. 1", section="безопасность",
            text=("Роли в пространстве: Admin, Developer, Viewer. MFA обязателен для "
                  "Admin и Developer; привязка происходит к SSO-аккаунту организации."),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-billing-hours",
    doc_num="05-Б",
    family_title="Оплата сверхлимитных часов",
    editions=(
        Edition(
            version=1, valid_from="2025-09-01", valid_to="2026-05-31",
            title="Сверхлимитные часы, ред. 1", section="биллинг",
            text=("Тариффикация за пределами нормы до 01.06.2026: 0.9 за час сверх лимита; "
                  "норма включена в подписку."),
        ),
        Edition(
            version=2, valid_from="2026-06-01", valid_to=None,
            title="Сверхлимитные часы, ред. 2", section="биллинг",
            text=("С июня 2026: ставка 1.1 за час сверх лимита, в подписку включены "
                  "100 бесплатных часов переработки в месяц."),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-monitoring-alerts",
    doc_num="20-Р",
    family_title="Мониторинг и каналы алертов",
    editions=(
        Edition(
            version=1, valid_from="2025-12-01", valid_to="2026-05-31",
            title="Каналы алертов, ред. 1", section="мониторинг",
            text=("До 01.06.2026 алерты отправляются только на email, "
                  "порог загрузки CPU — 85%."),
        ),
        Edition(
            version=2, valid_from="2026-06-01", valid_to=None,
            title="Каналы алертов, ред. 2", section="мониторинг",
            text=("С июня 2026 алерты уходят в push-уведомления приложения и Telegram, "
                  "порог CPU снижен до 80%; доступна интеграция webhook."),
        ),
    ),
))

register_seed(KnowledgeDoc(
    doc_id="doc-regions",
    doc_num="04-Р",
    family_title="Регионы и часовые пояса",
    editions=(
        Edition(
            version=1, valid_from="2026-01-01", valid_to=None,
            title="Регионы площадки, ред. 1", section="эксплуатация",
            text=("Доступные регионы: Москва, Санкт-Петербург, Новосибирск, Сочи, "
                  "Владивосток. Регион клиента влияет на плановые окна обслуживания."),
        ),
    ),
))

DEMO_QUERY_TARIF_LIMIT = "какой лимит параллельных воркеров был у тарифа Start?"
DEMO_QUERY_NUM = "покажи документ 06-Т"

AS_OF_DECEMBER = "2025-12-05"
AS_OF_JUNE = "2026-06-10"
