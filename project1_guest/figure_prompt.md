# Figure Prompt — Diffusion Determinants: Guest & Zeolite Perspectives (Methodology Figure)

Design a **single overview figure** that explains **how the analysis was structured**, not the findings. Target: JACS / Angew. Chem. style. Data from Project 1 (guest-centric) and Project 2 (zeolite-centric).

---

## Panel A — Analytical Framework Overview

Central schematic showing the **two symmetrical analysis routes**:

**Left route (Guest-centric)** — "Lock the molecule, vary the framework":  
Select top-20 guests by data abundance. For each guest, systematically vary → zeolite identity, topology, Si/Al ratio, exchangeable cation, temperature, loading. Measure effect on log₁₀(D) via 6 statistical methods.

**Right route (Zeolite-centric)** — "Lock the framework, vary the molecule":  
Select top-20 zeolites by data abundance. For each zeolite, systematically vary → guest identity, kinetic diameter, Si/Al ratio, exchangeable cation, temperature, loading.

Two routes converge at center: "Diffusion Determinants" (guest × zeolite interaction).

Draw as two mirrored flow diagrams meeting in the middle. Guest route in teal (#2c7bb6), zeolite route in coral (#d7191c).

- **Panel label**: `(a)`.

---

## Panel B — Statistical Methods Table

Compact visual table mapping each research question to its statistical method:

| Question | Method | Output Metric | Threshold |
|----------|--------|---------------|-----------|
| Does zeolite/guest identity matter? | Kruskal-Wallis H | η² (effect size), p-value | p < 0.05 |
| Does Si/Al ratio matter? | Spearman rank correlation | r, p-value | p < 0.05, n ≥ 5 |
| Does temperature matter? | Arrhenius linear regression (ln D vs 1/T) | Eₐ (kJ/mol), R² | n ≥ 5 points |
| Does ion exchange matter? | Mann-Whitney U (pairwise) | Δ mean logD, p-value | n ≥ 3 per group |
| Does loading/concentration matter? | Spearman rank correlation | r, p-value | n ≥ 5 |

Render as a clean table with method names in monospace, colored headers. Add a small icon or glyph per row to indicate the variable type (continuous/discrete).

- **Panel label**: `(b)`.

---

## Panel C — Data Coverage & Variable Space

Schematic showing the structure of the cleaned dataset (N = 3,431, 24 columns):

- **Left**: Hierarchical breakdown of categorical variables — guest_molecule (227) → std_zeolite_name (topologies: MFI, FAU, LTA, DDR, BEA, CHA, ...) → method_type (Experimental / Computational) → method_category (MD, PFG NMR, QENS, ZLC, Uptake, Membrane, ...)
- **Right**: Distribution of continuous variables — temperature_value (range, K), si_al_ratio (log-scale histogram), diffusion_coefficient_value (log-scale histogram spanning 10⁻¹⁴ to 10⁻⁶ m²/s)

Show as small-multiples: one categorical Sankey/alluvial on the left, three compact histograms on the right.

- **Panel label**: `(c)`.

---

## Panel D — Arrhenius & Si/Al Methodology

Two-panel inset explaining the key physical models:

**Left (Arrhenius)**: Mini schematic. Show the linearized equation `ln D = ln D₀ − (Eₐ/R)·(1/T)`, with a sketch of the regression line on 1000/T axes. Annotate: "Slope = −Eₐ/R". Note that this is fitted per-guest and per-zeolite for subsets with ≥ 5 (T, D) points.

**Right (Si/Al)**: Mini schematic. Show a scatter of logD vs Si/Al with a Spearman rank correlation line. Annotate: "Non-parametric, captures monotonic trends". Note that analysis is repeated within each topology subgroup when n ≥ 10.

- **Panel label**: `(d)`.

---

**Global specs**: 300 dpi, full-page width (~180 mm), #333333 text, Helvetica/Arial 8 pt, 0.5 pt spines, teal (#2c7bb6) + coral (#d7191c) + gray (#999999) palette. Export `.pdf` + `.png`.
