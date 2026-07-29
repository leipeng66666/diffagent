"""
Streamlit Web Application - DiffAgent
Run: streamlit run streamlit_app.py

API key protection:
  - Streamlit Cloud: set secrets in Settings → Secrets
  - Local: create .streamlit/secrets.toml (see secrets.toml.example)
  - Or: set OPENAI_API_KEY environment variable
"""
import streamlit as st
import pandas as pd
import os
import sys
import io

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DiffAgent",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""<style>
    .stChatMessage { word-wrap: break-word; }
    .stSpinner > div { padding: 2rem 0; }
    footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# API KEY — read BEFORE any internal imports
# ═══════════════════════════════════════════════════════════════
def load_api_settings():
    """Load API settings from secrets → .env → env vars. Sets os.environ and returns dict."""
    settings = {}

    # 1) Try Streamlit secrets
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        try:
            val = st.secrets[key]
            if val and "your-" not in val:
                settings[key] = val
        except (KeyError, FileNotFoundError):
            pass

    # 2) Try python-dotenv .env file
    if "OPENAI_API_KEY" not in settings:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    # 3) Fall back to os.environ
    defaults = {
        "OPENAI_API_KEY": "",
        "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
        "OPENAI_MODEL": "deepseek-v4-pro",
    }
    for key, default in defaults.items():
        if key not in settings:
            settings[key] = os.environ.get(key, default)

    # Set os.environ so pydantic BaseSettings and sub-modules can read them
    for key, val in settings.items():
        if val:
            os.environ[key] = val

    return settings

api_settings = load_api_settings()

# ═══════════════════════════════════════════════════════════════
# PATCH config.settings NOW (before any internal import)
# ═══════════════════════════════════════════════════════════════
from config import settings
settings.OPENAI_API_KEY = api_settings["OPENAI_API_KEY"]
settings.OPENAI_BASE_URL = api_settings["OPENAI_BASE_URL"]
settings.OPENAI_MODEL = api_settings["OPENAI_MODEL"]

# ═══════════════════════════════════════════════════════════════
# CACHED RESOURCES (one-time heavy init)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model & building knowledge graph…")
def get_table_agent():
    """Create TableAgent singleton. Cached across all sessions."""
    from table_agent import TableAgent
    return TableAgent()

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_data(agent, file_path: str) -> bool:
    """Load CSV into agent. Returns True on success."""
    result = agent.load_table(file_path)
    if result.get("success"):
        shape = f"{result.get('shape', ['?', '?'])[0]} rows × {result.get('shape', ['?', '?'])[1]} cols"
        st.session_state.data_shape = shape
        return True
    return False

def ensure_data_loaded(agent) -> bool:
    """Re-load data into agent if it was lost (e.g. after agent re-init)."""
    if agent.current_data is not None:
        return True
    src = st.session_state.get("data_source_path")
    if src and os.path.exists(src):
        return load_data(agent, src)
    return False

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════
for key, default in {
    "messages": [],            # chat history: list of {role, content}
    "data_loaded": False,
    "data_source_path": None,
    "data_name": None,
    "data_shape": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

BUILTIN_CSV = "data/consolidated_cleand.csv"
api_key = api_settings["OPENAI_API_KEY"]

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
        st.title("🧪 DiffAgent")

    # ── API status ──
    if api_key:
        masked = api_key[:4] + "···" + api_key[-4:] if len(api_key) > 8 else "****"
        st.success(f"🔑 {masked}")
    else:
        st.error("⚠️ No API key!")
        st.caption("Set `OPENAI_API_KEY` in `.streamlit/secrets.toml`")
    st.divider()

    # ── DATA SOURCE ──
    st.subheader("📂 Data Source")
    data_choice = st.radio(
        "Select data source",
        ["📦 Built-in CSV", "📤 Upload my own"],
        label_visibility="collapsed",
    )

    if data_choice == "📦 Built-in CSV":
        if os.path.exists(BUILTIN_CSV):
            size_mb = os.path.getsize(BUILTIN_CSV) / 1024 / 1024
            st.caption(f"`consolidated_cleand.csv` ({size_mb:.1f} MB)")
            if st.button("⚡ Load Data", use_container_width=True, type="primary"):
                with st.spinner("Building indexes & knowledge graph…"):
                    agent = get_table_agent()
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
        uploaded_file = st.file_uploader(
            "Choose CSV / Excel / JSON",
            type=["csv", "xlsx", "xls", "json"],
        )
        if uploaded_file is not None:
            st.caption(f"📎 `{uploaded_file.name}` ({uploaded_file.size / 1024:.0f} KB)")
            if st.button("⚡ Load Uploaded File", use_container_width=True, type="primary"):
                with st.spinner("Building indexes & knowledge graph…"):
                    agent = get_table_agent()
                    os.makedirs("uploads", exist_ok=True)
                    tmp = f"uploads/{uploaded_file.name}"
                    with open(tmp, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    if load_data(agent, tmp):
                        st.session_state.data_loaded = True
                        st.session_state.data_source_path = tmp
                        st.session_state.data_name = uploaded_file.name
                        st.rerun()
                    else:
                        st.error("Failed to load file.")

    st.divider()

    # ── STATUS ──
    if st.session_state.data_loaded:
        shape = st.session_state.get("data_shape", "? × ?")
        st.success(f"📊 {st.session_state.data_name}\n*{shape}*")
    else:
        st.info("📭 No data loaded")

    st.divider()

    # ── ACTIONS ──
    st.subheader("🧹 Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💣 Full Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            get_table_agent.clear()
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════
st.title("📊 DiffAgent")
st.caption(
    "Ask natural-language questions about molecular diffusion data. "
    "AI-powered ranking, comparison, and analysis for zeolite separation research."
)

# ── Data Preview ──
if st.session_state.data_loaded:
    agent = get_table_agent()
    with st.expander("🔍 Data Preview", expanded=False):
        try:
            preview = agent.get_data_preview(max_rows=20)
            if "error" not in preview:
                df_preview = pd.DataFrame(preview["data"], columns=preview["columns"])
                st.caption(f"First {min(20, len(df_preview))} rows of {preview['shape'][0]} total")
                st.dataframe(df_preview, use_container_width=True, height=250)
        except Exception as e:
            st.caption(f"Preview unavailable: {e}")

# ── Chat History ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("viz"):
            st.plotly_chart(msg["viz"], use_container_width=True)

# ── Chat Input ──
prompt_disabled = not st.session_state.data_loaded

chat_placeholder = (
    "e.g. 'Which zeolite is best for CH4/CO2 separation?' or "
    "'Compare methane and ethane diffusion in ZSM-5'"
)

if prompt := st.chat_input(placeholder=chat_placeholder, disabled=prompt_disabled):
    # Guard conditions
    if not st.session_state.data_loaded:
        st.warning("⚠️ Load data first (sidebar).")
        st.stop()
    if not api_key:
        st.error("🔐 No API key configured.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤔 *Analyzing data…*")

        try:
            agent = get_table_agent()
            ensure_data_loaded(agent)

            result = agent.process_query(prompt)

            if result.get("success"):
                answer = result["response"].get("answer", "(no answer)")
                thinking_placeholder.empty()
                st.markdown(answer)

                # Details dropdown
                with st.expander("📊 Details"):
                    route = result.get("method_used", "qa")
                    tokens = result["response"].get("tokens_used", "?")
                    mdl = result["response"].get("model", "?")
                    st.caption(f"Route: **{route}** | Model: `{mdl}` | Tokens: {tokens}")
                    if result.get("graph_info"):
                        st.json(result["graph_info"])

                msg = {"role": "assistant", "content": answer}

                # Plotly figures
                for viz in result.get("visualizations", []):
                    if viz.get("type") == "plotly" and viz.get("figure"):
                        st.plotly_chart(viz["figure"], use_container_width=True)
                        msg["viz"] = viz["figure"]

            else:
                err = result.get("message", "Unknown error")
                thinking_placeholder.empty()
                st.error(f"❌ {err}")
                msg = {"role": "assistant", "content": f"❌ *{err}*"}

        except Exception as e:
            import traceback
            thinking_placeholder.empty()
            st.error(str(e))
            with st.expander("🔍 Debug"):
                st.code(traceback.format_exc())
            msg = {"role": "assistant", "content": f"❌ **Error:** {str(e)}"}

        st.session_state.messages.append(msg)

# ── Footer ──
st.divider()
st.caption(
    "💡 **Examples:** 'Which zeolite is best for CH4/CO2 separation?' | "
    "'推荐分离CO2和N2最好的分子筛' | "
    "'Show me all data above 300K for methane'"
)
