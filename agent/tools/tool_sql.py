from agent.tools.registry import tool
from agent.llm_client import get_llm_client
from postgres_db import get_schema, run_query
from config import LLM_MODEL


def _generate_sql(user_question: str, schema: str) -> str:
    """
    Ask the LLM to convert a natural language question into a SQL query
    using the real DB schema.
    """
    client = get_llm_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert SQL generator for PostgreSQL.\n"
                    "Given a database schema and a user question, "
                    "return ONLY a valid SQL SELECT query — no explanation, "
                    "no markdown, no backticks, just raw SQL.\n\n"
                    "Rules:\n"
                    "- Only generate SELECT statements\n"
                    "- Use exact table and column names from the schema\n"
                    "- Add LIMIT 50 unless the user asks for all records\n"
                    "- If the question cannot be answered from the schema, "
                    "return: UNSUPPORTED\n\n"
                    f"Database schema:\n{schema}"
                )
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
    )

    return response.choices[0].message.content.strip()


def _format_results(rows: list[dict]) -> str:
    """Format query results into a readable markdown table."""
    if not rows:
        return "The query returned no results."

    headers = list(rows[0].keys())
    header_row = " | ".join(headers)
    separator = " | ".join(["---"] * len(headers))
    data_rows = [
        " | ".join(str(row.get(h, "")) for h in headers)
        for row in rows
    ]

    table = f"| {header_row} |\n| {separator} |\n"
    table += "\n".join([f"| {r} |" for r in data_rows])
    return table


@tool(
    description=(
        "Query the PostgreSQL database by converting a natural language "
        "question into SQL. Use this when the user asks about data, "
        "records, statistics, counts, or anything that sounds like it "
        "could come from a database — for example: 'how many orders were "
        "placed last month', 'show me all customers from Mumbai', "
        "'what is the total revenue', or when the user explicitly says "
        "'query the database' or 'from the db'."
    ),
    params={
        "question": {
            "type": "string",
            "description": "The user's natural language question to answer from the database"
        }
    },
    required=["question"]
)
def tool_sql(question: str, **kwargs) -> str:
    """
    Tool: Convert natural language to SQL, run it on PostgreSQL,
    and return formatted results.
    """
    try:
        # Step 1: Get live schema from DB
        schema = get_schema()

        # Step 2: Generate SQL from question + schema
        sql = _generate_sql(question, schema)

        # Step 3: Handle unsupported questions
        if sql.upper() == "UNSUPPORTED":
            return (
                "I couldn't find relevant data in the database to answer "
                "that question. Please check if the table or column exists."
            )

        # Step 4: Run the query
        rows = run_query(sql)

        # Step 5: Format and return
        result_table = _format_results(rows)
        return (
            f"**Generated SQL:**\n```sql\n{sql}\n```\n\n"
            f"**Results ({len(rows)} rows):**\n{result_table}"
        )

    except ValueError as e:
        return f"Query blocked: {str(e)}"
    except Exception as e:
        return f"Database error: {str(e)}"