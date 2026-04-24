import psycopg2
import psycopg2.extras


def get_conn(db_url: str = None, credentials: dict = None):
    """
    Create a PostgreSQL connection.
    Priority: credentials dict > db_url > env POSTGRES_URL
    """
    if credentials:
        return psycopg2.connect(
            host=credentials["host"],
            port=credentials.get("port", 5432),
            dbname=credentials["dbname"],
            user=credentials["user"],
            password=credentials["password"]
        )
    if db_url:
        return psycopg2.connect(db_url)

    # fallback to config
    from config import POSTGRES_URL
    return psycopg2.connect(POSTGRES_URL)


def get_schema(credentials: dict = None) -> str:
    """
    Fetch schema from DB.
    Returns string like: table(col TYPE, ...)
    """
    conn = get_conn(credentials=credentials)
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


def run_query(sql: str, credentials: dict = None) -> list[dict]:
    """
    Execute a SELECT query and return results as list of dicts.
    """
    sql_clean = sql.strip().rstrip(";")

    if not sql_clean.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    conn = get_conn(credentials=credentials)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_clean)
            rows = cur.fetchall()
        return [dict(row) for row in rows]
    except psycopg2.Error as e:
        raise ValueError(f"Query failed: {e}")
    finally:
        conn.close()


def test_connection(credentials: dict) -> tuple[bool, str]:
    """
    Test if DB credentials are valid.
    Returns (success: bool, message: str)
    """
    try:
        conn = get_conn(credentials=credentials)
        conn.close()
        return True, "Connection successful!"
    except psycopg2.OperationalError as e:
        return False, f"Connection failed: {str(e)}"