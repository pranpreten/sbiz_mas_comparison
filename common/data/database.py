"""매 조회마다 읽기 전용 트랜잭션을 사용한다. DDL/적재 기능 없음."""
from collections.abc import Mapping, Sequence
from typing import Any


class DatabaseError(RuntimeError):
    """접속 문자열이나 SQL 원문을 포함하지 않는 오류."""


class Database:
    def __init__(self, dsn: str, statement_timeout_ms: int = 10000, max_rows: int = 500):
        if not dsn.strip():
            raise ValueError("DATABASE_URL이 필요합니다.")
        if statement_timeout_ms < 1 or max_rows < 1:
            raise ValueError("DB 제한값은 양수여야 합니다.")
        self._dsn = dsn
        self.statement_timeout_ms = statement_timeout_ms
        self.max_rows = max_rows

    def __repr__(self):
        return f"Database(read_only=True, max_rows={self.max_rows})"

    def fetch_all(self, query: str, params: Sequence | Mapping = ()) -> list[dict[str, Any]]:
        import psycopg
        from psycopg.rows import dict_row

        # 이 API에는 개발자가 작성한 조회만 전달한다. 사용자 SQL을 받는 도구가 아니다.
        if not query.lstrip().upper().startswith(("SELECT", "WITH")):
            raise ValueError("조회 쿼리만 허용합니다.")
        try:
            with psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=5) as conn:
                conn.read_only = True
                conn.execute("SELECT set_config('statement_timeout', %s, true)",
                             (str(self.statement_timeout_ms),))
                conn.execute("SELECT set_config('lock_timeout', '2000', true)")
                rows = conn.execute(query, params).fetchmany(self.max_rows + 1)
                if len(rows) > self.max_rows:
                    raise DatabaseError("조회 결과 상한을 초과했습니다. 조건을 좁혀 주세요.")
                return rows
        except psycopg.Error:
            raise DatabaseError("DB 연결 또는 읽기 전용 조회에 실패했습니다.") from None

    def fetch_one(self, query: str, params: Sequence | Mapping = ()) -> dict | None:
        rows = self.fetch_all(query, params)
        if len(rows) > 1:
            raise DatabaseError("하나여야 하는 조회 결과가 여러 개입니다.")
        return rows[0] if rows else None
