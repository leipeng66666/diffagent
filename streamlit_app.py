"""DiffAgent v16 — Step 1: config + lightweight imports"""
import sys
import os

# Ensure local packages take precedence over site-packages (avoid "core" name conflict)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR in sys.path:
    sys.path.remove(_APP_DIR)
sys.path.insert(0, _APP_DIR)

import streamlit as st
import traceback

st.set_page_config(page_title="DiffAgent", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

st.title("🧪 DiffAgent v18")
st.success("✅ v18: LLM-mapped columns for diffusion coeff — accurate temperature pairing")

# Lightweight checks
_status = []

# pandas
try:
    import pandas as pd
    _status.append(("ok", f"pandas {pd.__version__}"))
except Exception as e:
    _status.append(("fail", f"pandas: {e}"))

# config
try:
    from config import settings
    _status.append(("ok", f"config (model={settings.OPENAI_MODEL})"))
except Exception as e:
    _status.append(("fail", f"config: {str(e)[:200]}"))
    class _FakeSettings:
        OPENAI_API_KEY = ""; OPENAI_BASE_URL = "https://api.deepseek.com/v1"; OPENAI_MODEL = "deepseek-v4-pro"
    settings = _FakeSettings()

# API settings
try:
    api_settings = {}
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        try:
            val = st.secrets[key]
            if val and "your-" not in val:
                api_settings[key] = val
        except (KeyError, FileNotFoundError):
            pass
    defaults = {"OPENAI_API_KEY": "", "OPENAI_BASE_URL": "https://api.deepseek.com/v1", "OPENAI_MODEL": "deepseek-v4-pro"}
    for key, default in defaults.items():
        if key not in api_settings:
            api_settings[key] = os.environ.get(key, default)
    for key, val in api_settings.items():
        if val:
            os.environ[key] = val
    settings.OPENAI_API_KEY = api_settings["OPENAI_API_KEY"]
    settings.OPENAI_BASE_URL = api_settings["OPENAI_BASE_URL"]
    settings.OPENAI_MODEL = api_settings["OPENAI_MODEL"]
    _status.append(("ok", f"API key={'✓' if api_settings['OPENAI_API_KEY'] else '✗'}"))
except Exception as e:
    _status.append(("fail", f"API: {e}"))
    api_settings = {"OPENAI_API_KEY": "", "OPENAI_BASE_URL": "", "OPENAI_MODEL": ""}

# Display
with st.expander("🔍 Status", expanded=True):
    fails = sum(1 for s, _ in _status if s == "fail")
    if fails == 0:
        st.success(f"All {len(_status)} checks passed ✅")
    else:
        st.error(f"{fails} FAILED")
    for s, msg in _status:
        st.success(msg) if s == "ok" else st.error(msg)

if fails > 0:
    st.stop()

st.divider()
st.caption("— v18: intelligent column mapping + accurate temperature pairing")

# ═══════════════════════════════════════════════════════════════
# FULL APP BELOW
# ═══════════════════════════════════════════════════════════════

st.markdown("""<style>
    .stChatMessage { word-wrap: break-word; }
    .stSpinner > div { padding: 2rem 0; }
    footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

_APP_VERSION = "v18"

@st.cache_resource(show_spinner="Loading embedding model & building knowledge graph…")
def get_table_agent(_cache_version: str):
    from table_agent import TableAgent
    return TableAgent()

def load_data(agent, file_path: str) -> bool:
    try:
        result = agent.load_table(file_path)
    except Exception as e:
        st.error(f"Loading crashed: {e}")
        st.code(traceback.format_exc())
        return False
    if not result.get("success"):
        st.error(f"Failed to load: {result.get('error', result.get('message', 'Unknown'))}")
        return False
    shape = f"{result['shape'][0]} rows × {result['shape'][1]} cols"
    st.session_state.data_shape = shape
    return True

for k, v in {"messages": [], "data_loaded": False, "data_source_path": None, "data_name": None, "data_shape": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

BUILTIN_CSV = "data/consolidated_cleand.csv"
api_key = api_settings["OPENAI_API_KEY"]

# ── Sidebar ──
with st.sidebar:
    st.title("🧪 DiffAgent")
    if api_key:
        st.success(f"🔑 {api_key[:4]}···{api_key[-4:]}" if len(api_key) > 8 else "🔑 ****")
    else:
        st.warning("⚠️ No API key")
    st.divider()

    st.subheader("📂 Data Source")
    data_choice = st.radio("Source", ["📦 Built-in CSV", "📤 Upload"], label_visibility="collapsed")

    if data_choice == "📦 Built-in CSV":
        if os.path.exists(BUILTIN_CSV):
            st.caption(f"`consolidated_cleand.csv` ({os.path.getsize(BUILTIN_CSV)/1024/1024:.1f} MB)")
            if st.button("⚡ Load Data", use_container_width=True, type="primary"):
                with st.spinner("Building indexes & knowledge graph…"):
                    agent = get_table_agent(_APP_VERSION)
                    if load_data(agent, BUILTIN_CSV):
                        st.session_state.data_loaded = True
                        st.session_state.data_source_path = BUILTIN_CSV
                        st.session_state.data_name = "Built-in CSV"
                        st.rerun()
                    else:
                        st.error("Failed to load.")
        else:
            st.warning(f"File not found:\n`{BUILTIN_CSV}`")
    else:
        uf = st.file_uploader("Choose CSV / Excel / JSON", type=["csv", "xlsx", "xls", "json"])
        if uf is not None:
            st.caption(f"📎 `{uf.name}` ({uf.size/1024:.0f} KB)")
            if st.button("⚡ Load Uploaded File", use_container_width=True, type="primary"):
                with st.spinner("Building indexes & knowledge graph…"):
                    agent = get_table_agent(_APP_VERSION)
                    os.makedirs("uploads", exist_ok=True)
                    tmp = f"uploads/{uf.name}"
                    with open(tmp, "wb") as f:
                        f.write(uf.getbuffer())
                    if load_data(agent, tmp):
                        st.session_state.data_loaded = True
                        st.session_state.data_source_path = tmp
                        st.session_state.data_name = uf.name
                        st.rerun()
                    else:
                        st.error("Failed to load.")

    st.divider()
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

# ── Main ──
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
if prompt := st.chat_input(placeholder="Ask about the data…", disabled=prompt_disabled):
    if not api_key:
        st.error("No API key")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ph = st.empty()
        ph.markdown("🤔 *Analyzing…*")
        try:
            agent = get_table_agent(_APP_VERSION)
            if agent.current_data is None:
                agent.load_table(st.session_state.data_source_path)
            result = agent.process_query(prompt)
            if result.get("success"):
                answer = result["response"].get("answer", "(no answer)")
                ph.markdown(answer)
                msg = {"role": "assistant", "content": answer}
                for viz in result.get("visualizations", []):
                    if viz.get("type") == "plotly" and viz.get("figure"):
                        st.plotly_chart(viz["figure"], use_container_width=True)
            else:
                ph.error(f"❌ {result.get('message', 'Unknown error')}")
                msg = {"role": "assistant", "content": f"❌ {result.get('message', '?')}"}
        except Exception as e:
            ph.error(str(e))
            with st.expander("Debug"):
                st.code(traceback.format_exc())
            msg = {"role": "assistant", "content": f"❌ {str(e)}"}
        st.session_state.messages.append(msg)

st.divider()
st.caption("💡 v17 — 'Which zeolite is best for CH4/CO2 separation?'")
