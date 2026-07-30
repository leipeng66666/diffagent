# -*- coding: utf-8 -*-
"""
End-to-end test of DiffAgent: QA, Ranking, and Visualization modes.
Run: python test_e2e.py
"""
import os, sys, io
sys.path.insert(0, '.')
sys.dont_write_bytecode = True
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["OPENAI_API_KEY"] = "sk-07230ef01ada4a7caa891eaa1ddf355a"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"
os.environ["OPENAI_MODEL"] = "deepseek-v4-pro"

from config import settings
settings.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
settings.OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
settings.OPENAI_MODEL = os.environ["OPENAI_MODEL"]

from table_agent import TableAgent

BUILTIN_CSV = "data/consolidated_cleand.csv"

def test_load():
    print("=" * 60)
    print("TEST 0: Data Loading")
    print("=" * 60)
    agent = TableAgent()
    result = agent.load_table(BUILTIN_CSV)
    if result.get("success"):
        s = result['shape']
        print(f"[OK] Loaded: {s[0]} rows x {s[1]} cols")
        return agent
    else:
        print(f"[FAIL] {result}")
        return None

def test_qa(agent):
    print()
    print("=" * 60)
    print("TEST 1: QA Mode")
    print("Query: 'What columns are in this dataset?'")
    print("=" * 60)
    result = agent.process_query("What columns are in this dataset?")
    if result.get("success"):
        ans = result["response"].get("answer", "")[:600]
        route = result.get("method_used", "?")
        tokens = result["response"].get("tokens_used", "?")
        print(f"[OK] Route: {route} | Tokens: {tokens}")
        print(f"Answer: {ans}")
    else:
        print(f"[FAIL] {result.get('message', result)}")

def test_ranking(agent):
    print()
    print("=" * 60)
    print("TEST 2: Ranking Mode")
    print("Query: 'Which zeolite is best for CO2/CH4 separation?'")
    print("=" * 60)
    result = agent.process_query("Which zeolite is best for CO2/CH4 separation?")
    if result.get("success"):
        ans = result["response"].get("answer", "")[:600]
        route = result.get("method_used", "?")
        tokens = result["response"].get("tokens_used", "?")
        print(f"[OK] Route: {route} | Tokens: {tokens}")
        print(f"Answer: {ans}")
    else:
        print(f"[FAIL] {result.get('message', result)}")

def test_viz(agent):
    print()
    print("=" * 60)
    print("TEST 3: Visualization")
    print("Query: '帮我绘制一个MFI的扩散系数活化能的图'")
    print("=" * 60)
    result = agent.process_query("帮我绘制一个MFI的扩散系数活化能的图")
    if result.get("success"):
        ans = result["response"].get("answer", "")[:600]
        route = result.get("method_used", "?")
        tokens = result["response"].get("tokens_used", "?")
        print(f"[OK] Route: {route} | Tokens: {tokens}")
        print(f"Answer: {ans}")
        viz = result.get("visualization")
        if viz and viz.get("image"):
            import base64
            img_data = viz["image"]
            print(f"[OK] Image: {len(img_data)} base64 chars")
            os.makedirs("test_output", exist_ok=True)
            path = "test_output/mfi_activation_energy.png"
            with open(path, "wb") as f:
                f.write(base64.b64decode(img_data))
            print(f"[OK] Saved to: {path}")
        else:
            print("[WARN] No image returned")
    else:
        print(f"[FAIL] {result.get('message', result)}")

if __name__ == "__main__":
    agent = test_load()
    if agent:
        test_qa(agent)
        test_ranking(agent)
        test_viz(agent)
        print()
        print("=" * 60)
        print("ALL TESTS DONE")
        print("=" * 60)
