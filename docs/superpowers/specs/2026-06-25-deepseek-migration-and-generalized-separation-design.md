# DeepSeek Migration & Generalized Molecular Separation Design

**Date**: 2026-06-25
**Status**: Approved
**Scope**: 8 files modified, ~200 lines added, ~150 lines changed, ~60 lines removed

---

## 1. Motivation

### Current State
- LLM backend is Alibaba Qwen (`qwen-max-latest`) via DashScope API
- Separation analysis is hardcoded for CH4/CO2 only — material detection, prompts, and table headers explicitly reference methane/carbon dioxide
- Column mapping relies on keyword matching (`if 'zeolite' in col.lower()`) rather than semantic understanding
- System cannot generalize to new CSV schemas or arbitrary molecule pairs

### Target State
- LLM backend: DeepSeek (`deepseek-pro`) via `https://api.deepseek.com/v1`
- Separation analysis works for **any two guest molecules** the user specifies
- Column mapping uses a **three-layer framework** (data type → LLM semantic inference → role-to-task mapping)
- **Dual-mode routing**: ranking mode (which zeolite is best?) and comparison mode (how different are X and Y on zeolite Z?)
- Self-test suite validates all changes before handoff

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Implementation approach | Incremental refactoring (Plan A) | Existing CH4/CO2 logic is battle-tested; replace hardcoded parts, preserve architecture |
| DeepSeek endpoint | `https://api.deepseek.com/v1` | Official API, OpenAI-compatible |
| Model | `deepseek-pro` | User-specified |
| Separation analysis scope | Dual-mode configurable | User can ask ranking ("best zeolite for X/Y?") or simple comparison ("how different on ZSM-5?") |
| Column mapping scope | Compromise (Plan C) | Preset data-science concepts (numeric/categorical/identifier) + LLM infers business semantics |

---

## 3. Architecture

```
┌──────────────────────────────────────────────┐
│                   app.py (FastAPI)            │
└────────────────────┬─────────────────────────┘
                     │
┌────────────────────▼─────────────────────────┐
│               table_agent.py                  │
│  ┌──────────────────────────────────────┐    │
│  │  Intent Classification (LLM)          │    │
│  │  Routes: ranking / comparison         │    │
│  └──────────┬───────────────────────────┘    │
│             │                                  │
│  ┌──────────▼───────────────────────────┐    │
│  │  IntelligentColumnMapper (3-Layer)    │    │
│  │  L1: Data Type → L2: LLM Semantic    │    │
│  │  → L3: Role→Task Mapping             │    │
│  └──────────┬───────────────────────────┘    │
│             │                                  │
│  ┌──────────▼───────────────────────────┐    │
│  │  Generalized Comparison Handler       │    │
│  │  - Dynamic molecule extraction        │    │
│  │  - Generic temperature pairing        │    │
│  │  - Dynamic column headers             │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

---

## 4. Section-by-Section Design

### 4.1 API Layer: Qwen → DeepSeek

**Files modified (4 files, 6 locations):**

1. **`config.py`** — Change defaults:
   - `OPENAI_API_KEY`: `your_api_key_here`
   - `OPENAI_BASE_URL`: `https://api.deepseek.com/v1`
   - `OPENAI_MODEL`: `deepseek-pro`

2. **`core/llm_integration.py`** (line 15-17) — Sync constructor defaults

3. **`core/llm_integration.py`** (line 41-65 system prompt) — Change "CH4 and CO2 diffusion coefficients" → dynamic "{mol1} and {mol2} diffusion coefficients"

4. **`.env` / `env_example.txt`** — Update example values

**Unchanged:** OpenAI SDK call pattern, retry logic, temperature=0, max_tokens settings.

---

### 4.2 Generalized Molecular Separation

**Files modified:**

#### 4.2.1 `table_agent.py::_handle_comparison_query`

- **Remove**: Hardcoded material detection for methane/CO2 only (lines 186-193)
- **Add**: `_extract_molecule_names(query)` — uses LLM to dynamically extract any guest molecule names
- **Remove**: Hardcoded `material_variants` dict (lines 971-976)
- **Add**: Dynamic variant generation from LLM-inferred molecule names + common aliases

#### 4.2.2 `table_agent.py::_filter_similar_temperature`

- **Before**: `material_variants` hardcodes only methane and CO2
- **After**: Accept `materials: List[str]` parameter, generate variants from the names themselves (lowercase, replace spaces, etc.)

#### 4.2.3 `table_agent.py::_generate_data_table`

- **Before**: Column headers hardcoded as `CH4_DiffCoef(m2/s)`, `CO2_DiffCoef(m2/s)` etc.
- **After**: Dynamic headers using `{mol1}_DiffCoef(m2/s)`, `{mol2}_DiffCoef(m2/s)` from actual molecule names

#### 4.2.4 `core/llm_integration.py` prompts

- **Before**: System prompt hardcodes "CH4/CO2 separation", "CH4 and CO2 diffusion coefficients"
- **After**: Dynamic injection of `{mol1}` and `{mol2}` into all comparison prompts
- **Add**: Separate prompt templates for ranking mode vs comparison mode

**Preserved unchanged:**
- Log10 ratio calculation algorithm (log10(D_max/D_min))
- Temperature pairing with ±20K tolerance
- Zeolite ranking by max Log10 ratio (descending)
- Smart data reduction (`_smart_reduce_data`)
- Relevance ranking (`_rank_by_relevance`)

---

### 4.3 Intelligent Column Mapping (3-Layer Framework)

**File modified**: `core/intelligent_column_mapper.py`

#### Layer 1 — Data Type Classification (rule-based, no LLM)
Scan each column's sample values and classify:
- **numeric**: int/float values → candidate for diffusion_coefficient, temperature, pressure, concentration, si_al_ratio, loading
- **categorical**: repeated string values → candidate for zeolite, guest_molecule, method, modified_ion
- **text**: unique/long strings → candidate for doi, distinguishing_variable, filename

#### Layer 2 — LLM Semantic Inference
Input: column name + first 5 sample values
Output: business role for each column

```
guest_molecule        → column containing molecule names
guest_composition     → column with molecular composition/formula
std_zeolite_name      → column with zeolite/material identifier
si_al_ratio           → column with Si/Al ratio values
modified_ion          → column with modifying ion/metal
loading_value         → column with metal/molecule loading amount
diffusion_coefficient → column with diffusion coefficient (numeric, m2/s)
temperature           → column with temperature (numeric, ~300-1000K)
pressure              → column with pressure (numeric)
concentration         → column with concentration (numeric)
method_category       → column with experimental method type
distinguishing_variable → column with differentiating factor
doi                   → column with DOI/reference
```

#### Layer 3 — Role-to-Task Mapping
Given analysis task requirements, retrieve column names by role:

```python
{
    "guest_col": "guest_molecule",       # actual column name mapped
    "zeolite_col": "std_zeolite_name",
    "diffusion_col": "diffusion_coefficient_value",
    "diffusion_unit_col": "diffusion_coefficient_unit",
    "temperature_col": "temperature_value",
    "pressure_col": "pressure_value",
    "concentration_col": "concentration_value",
    "method_col": "method_category",
    "doi_col": "doi",
    ...
}
```

**New output structure** of `map_query_to_columns()`:
```python
{
    "column_roles": { ... },        # role → actual column name
    "detected_molecules": [...],    # molecules found in data
    "detected_zeolites": [...],     # zeolites found in data (optional)
    "material_column": "...",       # column for guest molecule
    "zeolite_column": "...",        # column for zeolite name
    "temperature_column": "...",
    "concentration_column": "...",
    "method_column": "...",
    "confidence": 0.92
}
```

---

### 4.4 Dual-Mode Routing

**File modified**: `table_agent.py::process_query`

#### Intent Classification

LLM classifies query into one of two modes based on keywords and semantics:

| Mode | Triggers | Behavior |
|------|----------|----------|
| **ranking** | 最好, 推荐, 最强, 哪些, 排名, best, strongest, which zeolite, recommend | Full table scan → temperature pairing → Log10 sort → Top-N recommendation |
| **comparison** | 差多少, 对比, 比较, compare, difference, 和 (with 2 molecules) | Look at specified zeolite/conditions → direct diffusion coefficient comparison → ratio answer |

**Fallback logic**: If no trigger words detected:
- Default to comparison mode
- If `detected_molecules >= 2` AND no specific zeolite named → auto-upgrade to ranking mode

**Routing implementation**:
```python
def _classify_query_intent(self, query: str, detected_molecules: List[str], 
                           has_specific_zeolite: bool) -> str:
    """Returns 'ranking' or 'comparison'"""
    # LLM call with few-shot examples
    # Key decision factors: presence of ranking keywords, number of molecules, 
    # whether a specific zeolite is named
```

---

### 4.5 Self-Test Suite

Executed after all code changes. Each test runs against the live DeepSeek API.

| # | Test Case | Expected Result |
|---|-----------|----------------|
| 1 | `LLMIntegration().generate_response("Hello", "", "analysis")` | Returns response from `deepseek-pro`, model field shows `deepseek-pro` |
| 2 | Load `consolidated_results3_clean.csv` | Recognizes new columns: `guest_composition`, `si_al_ratio`, `modified_ion`, `loading_value`, `distinguishing_variable` |
| 3 | "Which zeolite is best for CH4/CO2 separation?" | Returns ranked zeolite list with Log10 ratios (regression test) |
| 4 | "Which zeolite is best for separating methane and ethane?" | Returns Log10-ranked zeolite recommendations for methane/ethane pair |
| 5 | "How different are methane and ethane diffusion coefficients in ZSM-5?" | Returns direct comparison with ratio, no ranking table |
| 6 | "推荐分离二氧化碳和氮气最好的分子筛" | Chinese query correctly identified, ranked results returned |
| 7 | Load `test_data.csv` (different column names) | Adaptive column mapping works, query returns valid answer |
| 8 | "温度300K以上的数据有多少条？" | Non-separation query still works correctly |

---

## 5. Files Changed (Summary)

| File | Change Type | Description |
|------|-------------|-------------|
| `config.py` | Modify (3 lines) | API key, base_url, model → DeepSeek |
| `core/llm_integration.py` | Modify (~30 lines) | Constructor defaults, prompts generalized |
| `core/intelligent_column_mapper.py` | Modify (~80 lines) | 3-layer mapping framework, new output structure |
| `table_agent.py` | Modify (~120 lines) | Molecule generalization, dual-mode routing, dynamic headers |
| `env_example.txt` | Modify | Sample values → DeepSeek |
| `QWEN_INTEGRATION.md` | Delete | Replaced by DEEPSEEK_INTEGRATION.md |
| `DEEPSEEK_INTEGRATION.md` | New | DeepSeek integration documentation |
| `README.md` | Modify | Update API references |

---

## 6. What Stays Unchanged

- FastAPI routes (`app.py`)
- Visualization engine (`core/visualization_engine.py`)
- RAG engine (`core/rag_engine.py`)
- GraphRAG engine (`core/graphrag_engine.py`)
- Semantic parser (`core/semantic_parser.py`)
- Synonym mapper (`core/synonym_mapper.py`)
- Data extractor (`core/data_extractor.py`)
- Unit recognizer, code generator, intelligent filter
- SimpleDataFrame data structure
- Log10 ratio calculation algorithm
- Temperature pairing algorithm (±20K tolerance)
- Smart data reduction logic
- Web UI templates and static files
