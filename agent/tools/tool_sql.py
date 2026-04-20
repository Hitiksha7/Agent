import json
from agent.tools.registry import tool
from agent.llm_client import get_llm_client
from postgres_db import get_schema, run_query
from config import LLM_MODEL


def _generate_queries(user_question: str, schema: str) -> list[dict]:
    """
    Step 1: LLM breaks the question into one or more SQL queries.
    Returns list of {"label": str, "sql": str}
    """
    client = get_llm_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert PostgreSQL query generator.\n\n"
                    "Given a user question and a database schema, decide how many "
                    "SQL queries are needed to fully answer the question.\n\n"
                    "Return ONLY a JSON array. Each item must have:\n"
                    "  - 'label': short title describing what this query fetches\n"
                    "  - 'sql': a valid PostgreSQL SELECT statement\n\n"
                    "Rules:\n"
                    "- Only SELECT statements allowed\n"
                    "- Only use these 3 tables: orders, customers, locations\n"
                    "- Join orders + customers ON orders.customer_id = customers.customer_id\n"
                    "- Join orders + locations ON orders.city = locations.city\n"
                    "- STRICTLY split into ONE query per metric — never combine\n"
                    "- Count how many metrics the user asked — return that many queries\n"
                    "- Example: 'total orders, top city, top segment' = exactly 3 queries\n"
                        "- Example: 'total orders and top city' = exactly 2 queries\n"
                        "- Add LIMIT 1 for top/best queries\n"
                        "- Add LIMIT 50 for list queries\n"
                        "- No LIMIT for count/aggregate queries\n"
                    "Table structure:\n"
                        "- orders: row_id, order_id, order_date, ship_date, ship_mode, customer_id, city\n"
                        "- customers: customer_id, customer_name, segment\n"
                        "- locations: city, country_region\n"
                        "- To get segment → JOIN orders + customers ON orders.customer_id = customers.customer_id\n"
                        "- To get country → JOIN orders + locations ON orders.city = locations.city\n\n"
                    "- If unanswerable from schema: "
                        "[{\"label\": \"unsupported\", \"sql\": \"UNSUPPORTED\"}]\n\n"
                        "Count the distinct metrics in the question and return exactly "
                        "that many query objects — no more, no less.\n\n"
                        "Return ONLY the JSON array — no markdown, no backticks, no explanation."
                )
            },
            {"role": "user", "content": user_question}
        ]
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model added them
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def _run_all_queries(queries: list[dict]) -> list[dict]:
    """
    Step 2: Execute each SQL query and collect results.
    Returns list of {"label": str, "sql": str, "rows": list, "error": str|None}
    """
    results = []
    for q in queries:
        label = q.get("label", "Result")
        sql = q.get("sql", "").strip()

        if sql.upper() == "UNSUPPORTED":
            results.append({
                "label": label,
                "sql": sql,
                "rows": [],
                "error": "unsupported"
            })
            continue

        try:
            rows = run_query(sql)
            results.append({
                "label": label,
                "sql": sql,
                "rows": rows,
                "error": None
            })
        except Exception as e:
            results.append({
                "label": label,
                "sql": sql,
                "rows": [],
                "error": str(e)
            })

    return results


def _combine_results(user_question: str, results: list[dict]) -> str:
    """
    Step 3: LLM combines all query results into one natural language answer.
    """
    client = get_llm_client()

    # Build a summary of all results to pass to LLM
    results_text = ""
    for r in results:
        results_text += f"\n### {r['label']}\n"
        results_text += f"SQL: {r['sql']}\n"
        if r["error"]:
            results_text += f"Error: {r['error']}\n"
        elif not r["rows"]:
            results_text += "Result: No rows returned.\n"
        else:
            results_text += f"Result: {json.dumps(r['rows'], default=str)}\n"

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful data analyst assistant.\n"
                    "You have executed SQL queries on a PostgreSQL database "
                    "and received the results below.\n"
                    "Combine all results into a single clear, concise answer "
                    "to the user's original question.\n"
                    "Present numbers and data in a readable way.\n"
                    "If any query failed, mention it briefly but still answer "
                    "from the successful ones."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {user_question}\n\n"
                    f"Query results:{results_text}"
                )
            }
        ]
    )

    return response.choices[0].message.content.strip()


def _format_raw_results(results: list[dict]) -> str:
    sections = []
    for r in results:
        if r["error"] == "unsupported":
            continue

        section = f"**{r['label']}**\n"
        section += f"```sql\n{r['sql']}\n```\n"

        if r["error"]:
            section += f"⚠️ Error: {r['error']}\n"
        elif not r["rows"]:
            section += "No results.\n"
        else:
            headers = list(r["rows"][0].keys())
            header_row = " | ".join(headers)
            separator = " | ".join(["---"] * len(headers))  # ✅ fixed
            data_rows = "\n".join(
                "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
                for row in r["rows"]
            )
            section += f"| {header_row} |\n| {separator} |\n{data_rows}\n"

        sections.append(section)

    return "\n---\n".join(sections)


@tool(
    description=(
        "Query the PostgreSQL database using natural language. "
        "Use this when the user asks about data, records, statistics, "
        "counts, totals, filters, or anything that could come from a database. "
        "Also use when the user explicitly says 'query the database', "
        "'from the db', or 'in the database'. "
        "This tool automatically handles questions that need multiple queries."
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
    try:
        # Step 1: Get live schema
        schema = get_schema()

        # Step 2: LLM generates queries
        queries = _generate_queries(question, schema)

        # Step 3: Check if unsupported
        if len(queries) == 1 and queries[0].get("sql", "").upper() == "UNSUPPORTED":
            return (
                "I couldn't find relevant data in the database to answer "
                "that question. Please check if the table or column exists."
            )

        # Step 4: Run all queries
        results = _run_all_queries(queries)

        # Step 5: LLM combines results into natural answer
        combined_answer = _combine_results(question, results)

        # Step 6: Format raw SQL + tables for transparency
        raw_details = _format_raw_results(results)

        return (
            f"{combined_answer}\n\n"
            f"---\n"
            f"**Query Details:**\n\n"
            f"{raw_details}"
        )

    except json.JSONDecodeError:
        return "Failed to parse SQL queries from the model. Please try rephrasing."
    except ValueError as e:
        return f"Query blocked: {str(e)}"
    except Exception as e:
        return f"Database error: {str(e)}"