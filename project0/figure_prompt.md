# Figure Prompt — Project 0: Zeolite Separation Prediction (Methodology Figure)

Design a **single schematic overview figure** that describes the **workflow and methodology**, not the results. Target journal: JACS / Angew. Chem. — clean, minimal, instructive.

---

## Panel A — Overall Workflow

Illustrate the 4-step pipeline as a horizontal flowchart with icons and minimal text:

**Step 1** `Input` → Two guest molecules (e.g., O₂ / N₂).  
**Step 2** `Retrieve` → Search consolidated diffusion database (N = 3,431 records, 227 guests, 163 zeolites) for direct and similar-guest evidence. Group records by (zeolite, modification) into EvidencePackages.  
**Step 3** `Enrich` → Augment each candidate with LLM-generated metadata: guest molecular properties (family, size, polarity, functional groups) and zeolite framework properties (topology, pore aperture, Si/Al interpretation, cation effects).  
**Step 4** `Score & Rank` → Combine evidence quality (direct/near-direct/similar-guest), diffusion ratio between target guests (per-condition median), modification relevance, and mechanism plausibility into a weighted score. Output ranked candidate list.

Use muted teal (#2c7bb6) for data steps, coral (#d7191c) for LLM steps. Keep arrows simple, text ≤ 12 words per box.

- **Panel label**: `(a)`.

---

## Panel B — Evidence Retrieval Strategy

Schematic tree showing the retrieval logic hierarchy:
- **Level 1**: Direct evidence — both guests co-occur in same (zeolite, modification) group → strongest signal
- **Level 2**: Near-direct — same zeolite, different conditions (T, method, ion) → moderate signal
- **Level 3**: Similar-guest — LLM-identified structural analogs of target guests found in same zeolite → weak signal, requires molecular similarity reasoning
- **Level 4**: Single-side — only one target guest has records → weakest defensible signal

Draw as a descending ladder or nested decision tree, with evidence strength (number of candidate zeolites typically found) decreasing downward. Mark the fallback threshold — below this level, the system uses heuristics instead of LLM reasoning.

- **Panel label**: `(b)`.

---

## Panel C — Scoring Framework

Illustrate the 4-dimensional scoring rubric as a radar/spider chart with one example candidate overlaid:

Axes: (1) Evidence Quality (0–5), (2) Diffusion Selectivity (0–5, based on per-condition median O₂/N₂ ratio), (3) Modification Relevance (0–2), (4) Mechanism Plausibility (0–3). Fill the candidate polygon in semi-transparent teal.

Below the radar, add a mini-table showing the score thresholds for the diffusion axis:
| Ratio | Score |
|-------|-------|
| > 50× | 5 |
| 10–50× | 4 |
| 3–10× | 3 |
| 1.3–3× | 2 |
| < 1.3× | 0 |

- **Panel label**: `(c)`.

---

**Global specs**: 300 dpi, 2-column width (~180 mm), #333333 text, Helvetica/Arial, 0.5 pt spines, teal+coral+gray palette. Export `.pdf` + `.png`.
