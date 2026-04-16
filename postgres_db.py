import psycopg2
import psycopg2.extras
from config import POSTGRES_URL


def get_conn():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(POSTGRES_URL)


def get_schema() -> str:
    """
    Inspect all tables and columns in the public schema.
    Returns a string like:
        table_name(col1 TYPE, col2 TYPE, ...)
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return "No tables found in public schema."

    tables = {}
    for table_name, column_name, data_type in rows:
        if table_name not in tables:
            tables[table_name] = []
        tables[table_name].append(f"{column_name} {data_type.upper()}")

    return "\n".join(
        f"{table}({', '.join(cols)})"
        for table, cols in tables.items()
    )


def run_query(sql: str) -> list[dict]:
    """
    Execute a SELECT query and return results as a list of dicts.
    """
    sql_clean = sql.strip().rstrip(";")

    if not sql_clean.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_clean)
            rows = cur.fetchall()
        return [dict(row) for row in rows]
    except psycopg2.Error as e:
        raise ValueError(f"Query failed: {e}")
    finally:
        conn.close()