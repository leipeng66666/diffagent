"""Test natural gas query routing with debug."""
import sys, os, json, io
os.environ.setdefault("OPENAI_API_KEY", "sk-07230ef01ada4a7caa891eaa1ddf355a")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
os.environ.setdefault("OPENAI_MODEL", "deepseek-v4-pro")

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR in sys.path:
    sys.path.remove(_APP_DIR)
sys.path.insert(0, _APP_DIR)

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from table_agent import TableAgent

agent = TableAgent()
agent.load_table("data/consolidated_cleand.csv")

# First, check what _understand_query returns directly
query = "Which zeolite exhibits the highest performance in natural gas purification?"
print(f"Query: {query}")
print()

# Direct LLM call
understanding = agent.column_mapper._understand_query(query)
print("=== LLM _understand_query result ===")
print(json.dumps(understanding, indent=2, ensure_ascii=False))
print()

# Now full process
print("=== process_query ===")
r = agent.process_query(query)
print(f"method: {r.get('method_used')}")
print(f"route from parsed_query: {r.get('parsed_query', {}).get('route', '?')}")
print(f"success: {r.get('success')}")

if r.get("success"):
    answer = r["response"].get("answer", "")
    with open("test_results/natural-gas-debug.txt", "w", encoding="utf-8") as f:
        f.write(answer)
    print(f"answer saved to test_results/natural-gas-debug.txt ({len(answer)} chars)")
