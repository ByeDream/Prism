"""SQLite-backed usage recorder for per-client request tracking."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    client        TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    stream        INTEGER DEFAULT 0,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    latency_ms    INTEGER DEFAULT 0,
    status        INTEGER DEFAULT 200,
    created_at    TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_req_client  ON requests(client);
CREATE INDEX IF NOT EXISTS idx_req_created ON requests(created_at);
"""


class UsageRecorder:
    """Thread-safe, async-friendly usage recorder backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.executescript(_SCHEMA)
        conn.close()

    # -- recording ---------------------------------------------------------

    def _record_sync(
        self,
        client: str,
        model: str,
        stream: bool,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        status: int,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO requests (client, model, stream, input_tokens, output_tokens, latency_ms, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (client, model, int(stream), input_tokens, output_tokens, latency_ms, status),
        )
        conn.commit()

    async def record(
        self,
        client: str,
        model: str,
        stream: bool,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        status: int,
    ) -> None:
        await asyncio.to_thread(
            self._record_sync, client, model, stream, input_tokens, output_tokens, latency_ms, status
        )

    # -- querying ----------------------------------------------------------

    def _query_sync(
        self,
        *,
        client: str | None = None,
        model: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        where, params = self._build_where(client, model, start, end)

        conn = self._get_conn()
        count_row = conn.execute(f"SELECT COUNT(*) FROM requests{where}", params).fetchone()
        total = count_row[0]

        rows = conn.execute(
            f"SELECT * FROM requests{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()

        return [dict(r) for r in rows], total

    async def query(
        self,
        *,
        client: str | None = None,
        model: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        return await asyncio.to_thread(
            self._query_sync, client=client, model=model, start=start, end=end, limit=limit, offset=offset
        )

    # -- summary -----------------------------------------------------------

    def _summary_sync(
        self,
        *,
        client: str | None = None,
        model: str | None = None,
        start: str | None = None,
        end: str | None = None,
        group_by: str | None = None,
    ) -> dict:
        where, params = self._build_where(client, model, start, end)

        conn = self._get_conn()

        totals_row = conn.execute(
            f"""SELECT COUNT(*) AS total_requests,
                       COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS total_output_tokens
                FROM requests{where}""",
            params,
        ).fetchone()

        result: dict = {
            "total_requests": totals_row["total_requests"],
            "total_input_tokens": totals_row["total_input_tokens"],
            "total_output_tokens": totals_row["total_output_tokens"],
        }

        if group_by in ("client", "model", "day"):
            group_expr = "date(created_at)" if group_by == "day" else group_by
            group_label = "day" if group_by == "day" else group_by
            rows = conn.execute(
                f"""SELECT {group_expr} AS grp,
                           COUNT(*) AS requests,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens
                    FROM requests{where}
                    GROUP BY grp ORDER BY grp""",
                params,
            ).fetchall()
            result["breakdown"] = [
                {
                    group_label: r["grp"],
                    "requests": r["requests"],
                    "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                }
                for r in rows
            ]

        return result

    async def summary(
        self,
        *,
        client: str | None = None,
        model: str | None = None,
        start: str | None = None,
        end: str | None = None,
        group_by: str | None = None,
    ) -> dict:
        return await asyncio.to_thread(
            self._summary_sync, client=client, model=model, start=start, end=end, group_by=group_by
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _build_where(
        client: str | None, model: str | None, start: str | None, end: str | None
    ) -> tuple[str, list]:
        clauses: list[str] = []
        params: list = []
        if client:
            clauses.append("client = ?")
            params.append(client)
        if model:
            clauses.append("model = ?")
            params.append(model)
        if start:
            clauses.append("created_at >= ?")
            params.append(start)
        if end:
            clauses.append("created_at <= ?")
            params.append(end)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params
