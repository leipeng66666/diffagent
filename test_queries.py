"""Run test queries against TableAgent - save output to files."""
import sys, os, json, traceback, io

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR in sys.path:
    sys.path.remove(_APP_DIR)
sys.path.insert(0, _APP_DIR)

os.environ.setdefault("OPENAI_API_KEY", "sk-07230ef01ada4a7caa891eaa1ddf355a")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
os.environ.setdefault("OPENAI_MODEL", "deepseek-v4-pro")

from table_agent import TableAgent

QUERIES = [
    ("ch4-co2", "Which zeolite exhibits stronger separation performance for methane and carbon dioxide at similar temperatures?"),
    ("co2-n2", "Which zeolite exhibits stronger separation performance for carbon dioxide and nitrogen at similar temperatures?"),
    ("n2-o2", "Which zeolite exhibits stronger separation performance for nitrogen and oxygen at similar temperatures?"),
    ("c3h8-c3h6", "Which zeolite exhibits stronger separation performance for propane and propylene at similar temperatures?"),
    ("h2o-ethanol", "Which zeolite exhibits stronger separation performance for water and ethanol at similar temperatures?"),
    ("natural-gas", "Which zeolite exhibits the highest performance in natural gas purification?"),
    ("p-xylene", "Which molecular sieve is most favorable for the selectivity of para-xylene?"),
    ("lead-ion", "Which molecular sieve is most likely to remove the lead ions from the wastewater?"),
]

OUT_DIR = "test_results"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("Loading TableAgent...")
    agent = TableAgent()
    result = agent.load_table("data/consolidated_cleand.csv")
    if not result.get("success"):
        print(f"FAILED to load data: {result}")
        return
    print(f"Data loaded: {result['shape']}")

    for qid, query in QUERIES:
        print(f"\n[{qid}] Running: {query[:80]}...")
        out = {"query": query}

        try:
            r = agent.process_query(query)
            if r.get("success"):
                answer = r["response"].get("answer", "")
                out["answer"] = answer
                out["method"] = r.get("method_used", "?")
                out["route"] = r.get("parsed_query", {}).get("route", "?")
                out["success"] = True
                print(f"  Method: {out['method']} | Route: {out['route']} | Answer length: {len(answer)}")
            else:
                out["success"] = False
                out["error"] = r.get("message", "Unknown")
                print(f"  FAILED: {out['error']}")
        except Exception as e:
            out["success"] = False
            out["error"] = str(e)
            out["traceback"] = traceback.format_exc()
            print(f"  EXCEPTION: {e}")

        # Save to file (UTF-8, no terminal encoding issues)
        fname = f"{OUT_DIR}/{qid}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {fname}")

    print("\nDONE. Results in test_results/")

if __name__ == "__main__":
    main()
