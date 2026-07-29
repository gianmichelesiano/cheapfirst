"""Metriche e logging su SQLite."""

import sqlite3
import time
from pathlib import Path
from typing import Optional


CREATE_REQUESTS = """
CREATE TABLE IF NOT EXISTS requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL DEFAULT (datetime('now')),
    task_type     TEXT NOT NULL,
    difficulty    REAL NOT NULL,
    confidence    REAL NOT NULL,
    model_used    TEXT NOT NULL,
    turns         INTEGER DEFAULT 1,
    verify_used   INTEGER DEFAULT 0,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd      REAL DEFAULT 0,
    latency_ms    INTEGER DEFAULT 0,
    success       INTEGER DEFAULT 1,
    error_msg     TEXT
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model_used);",
    "CREATE INDEX IF NOT EXISTS idx_requests_task ON requests(task_type);",
]


class MetricsLogger:
    """Logga le richieste su SQLite e genera report."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(CREATE_REQUESTS)
            for idx in CREATE_INDEXES:
                try:
                    conn.execute(idx)
                except sqlite3.OperationalError:
                    pass
            conn.commit()
        finally:
            conn.close()

    def log(self, entry: dict):
        """Salva un record di richiesta."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                """INSERT INTO requests
                   (task_type, difficulty, confidence, model_used, turns,
                    verify_used, input_tokens, output_tokens, cost_usd,
                    latency_ms, success, error_msg)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.get("task_type", "general"),
                    entry.get("difficulty", 0.5),
                    entry.get("confidence", 0.5),
                    entry.get("model_used", ""),
                    entry.get("turns", 1),
                    1 if entry.get("verify_used") else 0,
                    entry.get("input_tokens", 0),
                    entry.get("output_tokens", 0),
                    entry.get("cost_usd", 0),
                    entry.get("latency_ms", 0),
                    1 if entry.get("success", True) else 0,
                    entry.get("error_msg", None),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def report(self, days: int = 7) -> str:
        """Genera report testuale delle ultime N giorni."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """SELECT
                    date(timestamp) as giorno,
                    COUNT(*) as richieste,
                    ROUND(SUM(cost_usd), 4) as spesa_totale,
                    ROUND(AVG(cost_usd), 6) as costo_medio,
                    ROUND(AVG(latency_ms), 0) as latenza_media,
                    ROUND(CAST(SUM(success) AS FLOAT) / COUNT(*) * 100, 1) as success_rate
                   FROM requests
                   WHERE timestamp >= datetime('now', ? || ' days')
                   GROUP BY giorno
                   ORDER BY giorno""",
                (f"-{days}",),
            ).fetchall()

            # Modelli più usati
            models = conn.execute(
                """SELECT model_used, COUNT(*) as count,
                          ROUND(AVG(cost_usd), 6) as avg_cost
                   FROM requests
                   WHERE timestamp >= datetime('now', ? || ' days')
                   GROUP BY model_used
                   ORDER BY count DESC
                   LIMIT 5""",
                (f"-{days}",),
            ).fetchall()

            # Totali
            totals = conn.execute(
                """SELECT
                    COUNT(*) as tot,
                    ROUND(SUM(cost_usd), 4) as spesa,
                    ROUND(AVG(cost_usd), 6) as avg_cost
                   FROM requests
                   WHERE timestamp >= datetime('now', ? || ' days')""",
                (f"-{days}",),
            ).fetchone()

        finally:
            conn.close()

        if not totals or totals["tot"] == 0:
            return "Nessuna richiesta nei giorni specificati."

        lines = [
            f"📊 Report ultimi {days} giorni",
            "─" * 40,
            f"Totale richieste:    {totals['tot']}",
            f"Spesa totale:       ${totals['spesa']:.4f}",
            f"Costo medio:        ${totals['avg_cost']:.6f}/richiesta",
            f"Success rate:       {rows[0]['success_rate'] if rows else 'N/A'}%",
            "",
            "Giorno   Richieste   Spesa   Costo medio   Latenza",
            "─" * 50,
        ]

        for r in rows:
            lines.append(
                f"{r['giorno']}  {r['richieste']:>5}     "
                f"${r['spesa_totale']:.3f}  "
                f"${r['costo_medio']:.6f}  "
                f"{r['latenza_media']}ms"
            )

        if models:
            lines.extend(["", "Modelli più usati:"])
            for m in models:
                lines.append(f"  {m['model_used']}: {m['count']} richieste, ${m['avg_cost']:.6f}/req")

        return "\n".join(lines)
