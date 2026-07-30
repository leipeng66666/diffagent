"""
DiffAgent v13 — Minimal bootstrap to verify Streamlit rendering
"""
import sys
import os

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Ensure local packages take precedence over site-packages.
# On Streamlit Cloud, pip may install a conflicting "core" package,
# causing "from core.simple_dataframe import ..." to find the wrong one.
# This MUST run before any other local import.
# ═══════════════════════════════════════════════════════════════
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Remove and re-insert to ensure local dir is FIRST
sys.path = [p for p in sys.path if p != _APP_DIR]
sys.path.insert(0, _APP_DIR)

import streamlit as st
import traceback

# ── Page config ──
try:
    st.set_page_config(page_title="DiffAgent", page_icon="🧪")
except Exception:
    pass

# ── Render test ──
st.title("🧪 DiffAgent v14 — sys.path fix")
st.success("✅ Streamlit is rendering. Python: " + sys.version.split()[0])

# ── Step-by-step imports ──
steps = []

def step(name, code):
    try:
        exec(code)
        steps.append(("ok", name))
    except Exception as e:
        steps.append(("fail", f"{name}: {e}"))

step("pandas", "import pandas as pd")
step("config", "from config import settings")

# ── API settings ──
try:
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        try:
            val = st.secrets[key]
            if val and "your-" not in val:
                os.environ[key] = val
        except (KeyError, FileNotFoundError):
            pass
    if "OPENAI_API_KEY" not in os.environ:
        os.environ.setdefault("OPENAI_API_KEY", "")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    os.environ.setdefault("OPENAI_MODEL", "deepseek-v4-pro")
    steps.append(("ok", "API settings"))
except Exception as e:
    steps.append(("fail", f"API: {e}"))

# ── table_agent — SKIP at startup (too heavy: pulls in sentence-transformers, matplotlib)
# It will be imported lazily inside get_table_agent() when user clicks "Load Data"
steps.append(("skip", "table_agent (lazy — loaded when data is first loaded)"))

# ── Display ──
with st.expander("🔍 Status", expanded=True):
    fails = sum(1 for s, _ in steps if s == "fail")
    if fails == 0:
        st.success(f"All {len(steps)} checks passed ✅")
    else:
        st.error(f"{fails} FAILED ❌")
    for s, msg in steps:
        if s == "ok":
            st.success(msg)
        else:
            st.error(msg)

if fails > 0:
    st.stop()

# ── Full app (simplified) ──
st.divider()
st.caption("— v14: sys.path fix for core package conflict")

import pandas as pd

_APP_VERSION = "v14"

@st.cache_resource(show_spinner="Loading…")
def get_table_agent(_v: str):
    from table_agent import TableAgent
    return TableAgent()

api_key = os.environ.get("OPENAI_API_KEY", "")

# Session state
for k, v in {"messages": [], "data_loaded": False, "data_source_path": None, "data_name": None, "data_shape": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

BUILTIN_CSV = "data/consolidated_cleand.csv"

# Sidebar
with st.sidebar:
    st.title("🧪 DiffAgent")
    if api_key:
        st.success(f"🔑 {api_key[:4]}···{api_key[-4:]}" if len(api_key) > 8 else "🔑 ****")
    else:
        st.warning("⚠️ No API key")
    st.divider()

    st.subheader("📂 Data")
    data_choice = st.radio("Source", ["📦 Built-in CSV", "📤 Upload"], label_visibility="collapsed")

    if data_choice == "📦 Built-in CSV":
        if os.path.exists(BUILTIN_CSV):
            st.caption(f"{os.path.getsize(BUILTIN_CSV)/1024/1024:.1f} MB")
            if st.button("⚡ Load Data", use_container_width=True, type="primary"):
                with st.spinner("Loading…"):
                    agent = get_table_agent(_APP_VERSION)
                    try:
                        result = agent.load_table(BUILTIN_CSV)
                        if result.get("success"):
                            st.session_state.data_loaded = True
                            st.session_state.data_source_path = BUILTIN_CSV
                            st.session_state.data_name = "Built-in CSV"
                            st.session_state.data_shape = f"{result['shape'][0]} rows × {result['shape'][1]} cols"
                            st.rerun()
                        else:
                            st.error(f"Failed: {result.get('error', '?')}")
                    except Exception as e:
                        st.error(f"Crashed: {e}")
                        st.code(traceback.format_exc())
        else:
            st.warning("File not found")

    if st.session_state.data_loaded:
        st.success(f"📊 {st.session_state.data_name}\n*{st.session_state.data_shape}*")
    else:
        st.info("📭 No data loaded")

    st.divider()
    if st.button("💣 Full Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        get_table_agent.clear()
        st.rerun()

# Main
if st.session_state.data_loaded:
    agent = get_table_agent(_APP_VERSION)
    with st.expander("🔍 Data Preview", expanded=False):
        try:
            preview = agent.get_data_preview(max_rows=20)
            if "error" not in preview:
                df_preview = pd.DataFrame(preview["data"], columns=preview["columns"])
                st.dataframe(df_preview, use_container_width=True, height=250)
        except Exception as e:
            st.caption(f"Preview unavailable: {e}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt_disabled = not st.session_state.data_loaded
if prompt := st.chat_input(placeholder="Ask a question about the data…", disabled=prompt_disabled):
    if not api_key:
        st.error("No API key")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 *Analyzing…*")
        try:
            agent = get_table_agent(_APP_VERSION)
            if agent.current_data is None:
                agent.load_table(st.session_state.data_source_path)
            result = agent.process_query(prompt)
            if result.get("success"):
                answer = result["response"].get("answer", "(no answer)")
                placeholder.markdown(answer)
                msg = {"role": "assistant", "content": answer}
                for viz in result.get("visualizations", []):
                    if viz.get("type") == "plotly" and viz.get("figure"):
                        st.plotly_chart(viz["figure"], use_container_width=True)
            else:
                placeholder.error(f"❌ {result.get('message', 'Unknown error')}")
                msg = {"role": "assistant", "content": f"❌ {result.get('message', '?')}"}
        except Exception as e:
            placeholder.error(str(e))
            with st.expander("Debug"):
                st.code(traceback.format_exc())
            msg = {"role": "assistant", "content": f"❌ {str(e)}"}
        st.session_state.messages.append(msg)

st.divider()
st.caption("💡 v14 — 'Which zeolite is best for CH4/CO2 separation?'")
