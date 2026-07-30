"""
Streamlit Web Application - DiffAgent
Run: streamlit run streamlit_app.py

API key protection:
  - Streamlit Cloud: set secrets in Settings → Secrets
  - Local: create .streamlit/secrets.toml (see secrets.toml.example)
  - Or: set OPENAI_API_KEY environment variable
"""
import streamlit as st
import traceback

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG — MUST be first Streamlit command
# ═══════════════════════════════════════════════════════════════
try:
    st.set_page_config(
        page_title="DiffAgent",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════
# IMMEDIATE VISIBLE OUTPUT
# ═══════════════════════════════════════════════════════════════
st.title("🧪 DiffAgent v12")
st.success("✅ Page loaded. Starting lightweight checks…")

# ═══════════════════════════════════════════════════════════════
# STEP 1: stdlib + pandas (all lightweight)
# ═══════════════════════════════════════════════════════════════
_status = []

try:
    import pandas as pd
    _status.append(("ok", f"pandas {pd.__version__}"))
except Exception as e:
    _status.append(("fail", f"pandas: {e}"))

try:
    import os, sys, io
    _status.append(("ok", "os, sys, io"))
except Exception as e:
    _status.append(("fail", f"stdlib: {e}"))

# ═══════════════════════════════════════════════════════════════
# STEP 2: config.settings
# ═══════════════════════════════════════════════════════════════
try:
    from config import settings
    _status.append(("ok", f"config.settings (model={settings.OPENAI_MODEL})"))
except Exception as e:
    _status.append(("fail", f"config.settings: {e}"))
    # Create a minimal fallback
    class _FallbackSettings:
        OPENAI_API_KEY = ""
        OPENAI_BASE_URL = "https://api.deepseek.com/v1"
        OPENAI_MODEL = "deepseek-v4-pro"
    settings = _FallbackSettings()

# ═══════════════════════════════════════════════════════════════
# STEP 3: API settings
# ═══════════════════════════════════════════════════════════════
try:
    def load_api_settings():
        s = {}
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
            try:
                val = st.secrets[key]
                if val and "your-" not in val:
                    s[key] = val
            except (KeyError, FileNotFoundError):
                pass
        if "OPENAI_API_KEY" not in s:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
        defaults = {
            "OPENAI_API_KEY": "",
            "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
            "OPENAI_MODEL": "deepseek-v4-pro",
        }
        for key, default in defaults.items():
            if key not in s:
                s[key] = os.environ.get(key, default)
        for key, val in s.items():
            if val:
                os.environ[key] = val
        return s

    api_settings = load_api_settings()
    settings.OPENAI_API_KEY = api_settings["OPENAI_API_KEY"]
    settings.OPENAI_BASE_URL = api_settings["OPENAI_BASE_URL"]
    settings.OPENAI_MODEL = api_settings["OPENAI_MODEL"]
    _status.append(("ok", f"API key={'✓' if api_settings['OPENAI_API_KEY'] else '✗ (set in secrets)'}"))
except Exception as e:
    _status.append(("fail", f"API settings: {e}"))
    api_settings = {"OPENAI_API_KEY": "", "OPENAI_BASE_URL": "", "OPENAI_MODEL": ""}

# ═══════════════════════════════════════════════════════════════
# Display status
# ═══════════════════════════════════════════════════════════════
with st.expander("🔍 Startup Diagnostics", expanded=True):
    fail_count = sum(1 for s, _ in _status if s == "fail")
    if fail_count == 0:
        st.success(f"All {len(_status)} checks passed ✅")
    else:
        st.error(f"{len(_status) - fail_count} passed, {fail_count} FAILED ❌")
    for status, msg in _status:
        if status == "ok":
            st.success(msg)
        else:
            st.error(msg)
    st.info(
        "Heavy modules (sentence-transformers, matplotlib, chromadb) "
        "are loaded lazily when data is first loaded — this is normal."
    )

if fail_count > 0:
    st.error("🛑 Stopped due to failures above.")
    st.stop()

st.divider()
st.caption("— *v12: lazy imports + startup diagnostics*")

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""<style>
    .stChatMessage { word-wrap: break-word; }
    .stSpinner > div { padding: 2rem 0; }
    footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CACHED RESOURCES — heavy imports happen INSIDE here (lazy!)
# ═══════════════════════════════════════════════════════════════
_APP_VERSION = "v12"

@st.cache_resource(show_spinner="Loading embedding model & building knowledge graph…")
def get_table_agent(_cache_version: str):
    """Create TableAgent singleton. Cached per-version.
    All heavy imports (sentence-transformers, matplotlib, etc.)
    happen here — NOT at module level."""
    from table_agent import TableAgent
    return TableAgent()

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_data(agent, file_path: str) -> bool:
    """Load CSV into agent. Returns True on success."""
    try:
        result = agent.load_table(file_path)
    except Exception as e:
        st.error(f"Loading crashed: {e}")
        st.code(traceback.format_exc())
        return False
    if not result.get("success"):
        err = result.get("error") or result.get("message", "Unknown error")
        st.error(f"Failed to load: {err}")
        err_tb = result.get("traceback", "")
        if err_tb:
            st.code(err_tb)
        else:
            st.caption("_(No traceback captured — full traceback will appear after next deployment)_")
        return False
    shape = f"{result.get('shape', ['?', '?'])[0]} rows × {result.get('shape', ['?', '?'])[1]} cols"
    st.session_state.data_shape = shape
    return True

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
    "messages": [],
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
try:
    with st.sidebar:
        st.title("🧪 DiffAgent")

        if api_key:
            masked = api_key[:4] + "···" + api_key[-4:] if len(api_key) > 8 else "****"
            st.success(f"🔑 {masked}")
        else:
            st.error("⚠️ No API key!")
            st.caption("Set `OPENAI_API_KEY` in `.streamlit/secrets.toml`")
        st.divider()

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
            uploaded_file = st.file_uploader(
                "Choose CSV / Excel / JSON",
                type=["csv", "xlsx", "xls", "json"],
            )
            if uploaded_file is not None:
                st.caption(f"📎 `{uploaded_file.name}` ({uploaded_file.size / 1024:.0f} KB)")
                if st.button("⚡ Load Uploaded File", use_container_width=True, type="primary"):
                    with st.spinner("Building indexes & knowledge graph…"):
                        agent = get_table_agent(_APP_VERSION)
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

        if st.session_state.data_loaded:
            shape = st.session_state.get("data_shape", "? × ?")
            st.success(f"📊 {st.session_state.data_name}\n*{shape}*")
        else:
            st.info("📭 No data loaded")

        st.divider()

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
except Exception as e:
    st.error(f"❌ Sidebar failed: {e}")
    st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════
try:
    if st.session_state.data_loaded:
        agent = get_table_agent(_APP_VERSION)
        with st.expander("🔍 Data Preview", expanded=False):
            try:
                preview = agent.get_data_preview(max_rows=20)
                if "error" not in preview:
                    df_preview = pd.DataFrame(preview["data"], columns=preview["columns"])
                    st.caption(f"First {min(20, len(df_preview))} rows of {preview['shape'][0]} total")
                    st.dataframe(df_preview, use_container_width=True, height=250)
            except Exception as e:
                st.caption(f"Preview unavailable: {e}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("viz"):
                st.plotly_chart(msg["viz"], use_container_width=True)

    prompt_disabled = not st.session_state.data_loaded
    chat_placeholder = (
        "e.g. 'Which zeolite is best for CH4/CO2 separation?' or "
        "'Compare methane and ethane diffusion in ZSM-5'"
    )

    if prompt := st.chat_input(placeholder=chat_placeholder, disabled=prompt_disabled):
        if not st.session_state.data_loaded:
            st.warning("⚠️ Load data first (sidebar).")
            st.stop()
        if not api_key:
            st.error("🔐 No API key configured.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🤔 *Analyzing data…*")

            try:
                agent = get_table_agent(_APP_VERSION)
                ensure_data_loaded(agent)

                result = agent.process_query(prompt)

                if result.get("success"):
                    answer = result["response"].get("answer", "(no answer)")
                    thinking_placeholder.empty()
                    st.markdown(answer)

                    with st.expander("📊 Details"):
                        route = result.get("method_used", "qa")
                        tokens = result["response"].get("tokens_used", "?")
                        mdl = result["response"].get("model", "?")
                        st.caption(f"Route: **{route}** | Model: `{mdl}` | Tokens: {tokens}")
                        if result.get("graph_info"):
                            st.json(result["graph_info"])

                    msg = {"role": "assistant", "content": answer}

                    for viz in result.get("visualizations", []):
                        if viz.get("type") == "plotly" and viz.get("figure"):
                            st.plotly_chart(viz["figure"], use_container_width=True)
                            msg["viz"] = viz["figure"]

                    viz_single = result.get("visualization")
                    if viz_single and viz_single.get("image"):
                        st.image(
                            f"data:image/{viz_single.get('format', 'png')};base64,{viz_single['image']}",
                            caption=viz_single.get("title", ""),
                            use_container_width=True,
                        )

                else:
                    err = result.get("message", "Unknown error")
                    thinking_placeholder.empty()
                    st.error(f"❌ {err}")
                    msg = {"role": "assistant", "content": f"❌ *{err}*"}

            except Exception as e:
                thinking_placeholder.empty()
                st.error(str(e))
                with st.expander("🔍 Debug"):
                    st.code(traceback.format_exc())
                msg = {"role": "assistant", "content": f"❌ **Error:** {str(e)}"}

            st.session_state.messages.append(msg)

except Exception as e:
    st.error(f"❌ Main area failed: {e}")
    st.code(traceback.format_exc())

st.divider()
st.caption(
    "💡 **Examples:** 'Which zeolite is best for CH4/CO2 separation?' | "
    "'推荐分离CO2和N2最好的分子筛' | "
    "'Show me all data above 300K for methane'"
)
