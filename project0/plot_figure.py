"""
Generate Project 0 methodology figure.
3-panel: (a) 4-step workflow, (b) evidence hierarchy, (c) scoring framework.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

# ── Global style ────────────────────────────────────────────
plt.rcParams.update({
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.unicode_minus": False,
    "text.color": "#333333",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.5,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})
TEAL = "#2c7bb6"
CORAL = "#d7191c"
GRAY = "#999999"
DARK = "#333333"
LIGHT_GRAY = "#f0f0f0"
WHITE = "#ffffff"


def draw_box(ax, x, y, w, h, text, color=TEAL, fontsize=8, bold=False,
             text_color=WHITE, edge_color=None, linewidth=1.0):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor=edge_color or color,
                         linewidth=linewidth, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, weight=weight, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.0, style="simple"):
    """Draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle="arc3,rad=0"))


def draw_label(ax, x, y, text, fontsize=7, color=DARK, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color)


# ═══════════════════════════════════════════════════════════════
# MAIN FIGURE
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12, 5.5), dpi=300)

# ── Panel (a): 4-step workflow ──────────────────────────────
ax_a = fig.add_axes([0.03, 0.55, 0.94, 0.42])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 4)
ax_a.axis("off")
ax_a.text(0.05, 3.85, "(a)  Workflow: Few-shot Zeolite Separation Prediction",
          fontsize=9, color=DARK, weight="bold")

# Steps
steps = [
    ("Input\nGuest Pair", "O2 / N2", TEAL),
    ("Retrieve\nEvidence", "Search DB\n(3,431 records)\nGroup by zeolite+ion", TEAL),
    ("Enrich\nMetadata", "LLM: guest & zeolite\nproperties", CORAL),
    ("Score & Rank\nCandidates", "Weighted 4-D score\n→ ranked list", TEAL),
]

box_w, box_h = 1.5, 1.8
y_center = 2.2
x_positions = [1.2, 3.1, 5.0, 7.8]

for i, (title, desc, color) in enumerate(steps):
    x = x_positions[i]
    # Main step box
    draw_box(ax_a, x, y_center + 0.4, box_w, 0.7, title, color=color,
             fontsize=7.5, bold=True, text_color=WHITE)
    # Description below
    draw_box(ax_a, x, y_center - 0.7, box_w, 1.0, desc,
             color=LIGHT_GRAY, fontsize=6.5, text_color=DARK,
             edge_color=GRAY, linewidth=0.5)

# Arrows between steps
for i in range(3):
    draw_arrow(ax_a, x_positions[i] + box_w/2 + 0.1, y_center + 0.4,
               x_positions[i+1] - box_w/2 - 0.1, y_center + 0.4, color=DARK)

# Step numbers above boxes
for i, x in enumerate(x_positions):
    ax_a.text(x, y_center + 1.3, f"Step {i+1}", ha="center", fontsize=7,
              color=DARK, weight="bold")

# Key data below
draw_label(ax_a, 4.5, 0.3,
           "Database: 227 guests · 163 zeolites · 24 columns · 5 evidence types · 4 scoring dimensions",
           fontsize=6.5, color=GRAY)
ax_a.text(0.05, 3.55, "(a)", fontsize=10, color=DARK, weight="bold")

# ── Panel (b): Evidence hierarchy ────────────────────────────
ax_b = fig.add_axes([0.03, 0.04, 0.44, 0.48])
ax_b.set_xlim(0, 7)
ax_b.set_ylim(0, 5.5)
ax_b.axis("off")
ax_b.text(0.05, 5.35, "(b)  Evidence Retrieval Hierarchy",
          fontsize=9, color=DARK, weight="bold")

levels = [
    ("Level 1 — Direct", "Both guests in same (zeolite, ion) group", 4.8, TEAL, "strongest"),
    ("Level 2 — Near-direct", "Same zeolite, different T / method / ion", 3.8, "#5ba3d9", "moderate"),
    ("Level 3 — Similar-guest", "Structural analogs via LLM similarity", 2.8, CORAL, "weak"),
    ("Level 4 — Single-side", "Only one target guest has records", 1.8, "#e58283", "weakest"),
]

box_w2, box_h2 = 5.5, 0.7
for title, desc, y, color, strength in levels:
    draw_box(ax_b, 3.3, y, box_w2, box_h2, "", color=LIGHT_GRAY,
             edge_color=color, linewidth=1.2)
    ax_b.text(0.35, y + 0.15, title, fontsize=7.5, color=color, weight="bold")
    ax_b.text(0.35, y - 0.18, desc, fontsize=6.5, color=DARK)
    ax_b.text(6.2, y, strength, fontsize=6, color=color, ha="center",
              style="italic")

# Downward arrows between levels
for i in range(3):
    y_top = levels[i][2] - box_h2/2
    y_bot = levels[i+1][2] + box_h2/2
    draw_arrow(ax_b, 3.3, y_top, 3.3, y_bot, color=GRAY, lw=0.8)

# Fallback annotation
ax_b.annotate("below this line →\nheuristic fallback",
              xy=(6.0, 2.3), fontsize=6, color=GRAY, ha="center",
              style="italic")
ax_b.axhline(y=2.3, xmin=0.1, xmax=0.85, color=GRAY, ls="--", lw=0.5)

# ── Panel (c): Scoring framework ────────────────────────────
ax_c_polar = fig.add_axes([0.55, 0.08, 0.42, 0.42], projection="polar")
ax_c_polar.set_facecolor("none")

categories = ["Evidence\nQuality", "Diffusion\nSelectivity",
              "Modification\nRelevance", "Mechanism\nPlausibility"]
max_vals = [5, 5, 2, 3]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # close

# Draw radar background
for level in [1, 2, 3, 4, 5]:
    vals = [level] * N + [level]
    ax_c_polar.fill(angles, vals, alpha=0, edgecolor="#e0e0e0", lw=0.3)

ax_c_polar.set_xticks(angles[:-1])
ax_c_polar.set_xticklabels(categories, fontsize=6.5, color=DARK)
ax_c_polar.set_ylim(0, 5.5)
ax_c_polar.set_yticks([1, 2, 3, 4, 5])
ax_c_polar.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=5.5, color=GRAY)
ax_c_polar.tick_params(pad=2)

# Example candidate (filled radar)
example_vals = [4, 3, 2, 3]
example_vals += example_vals[:1]
ax_c_polar.fill(angles, example_vals, alpha=0.25, color=TEAL)
ax_c_polar.plot(angles, example_vals, color=TEAL, lw=1.2)

# Legend
ax_c_polar.text(0.5, -0.22, "(c)  Scoring Framework — 4-dimension weighted evaluation",
                transform=ax_c_polar.transAxes, fontsize=8, color=DARK,
                weight="bold", ha="center")

# Threshold table below radar
ax_table = fig.add_axes([0.55, 0.01, 0.42, 0.08])
ax_table.axis("off")
table_data = [
    ["Diffusion Ratio", "< 1.3×", "1.3–3×", "3–10×", "10–50×", "> 50×"],
    ["Score", "0", "2", "3", "4", "5"],
]
table = ax_table.table(cellText=table_data, cellLoc="center",
                       loc="center", edges="horizontal")
table.auto_set_font_size(False)
table.set_fontsize(6)
for key, cell in table.get_celld().items():
    cell.set_edgecolor(GRAY)
    cell.set_linewidth(0.3)
    if key[0] == 0:  # header
        cell.set_text_props(weight="bold", color=DARK)
    if key[1] == 3:  # highlight "3–10×"
        cell.set_facecolor(f"{TEAL}20")

# ── Save ─────────────────────────────────────────────────────
output_path = "project0/output/methodology_figure.png"
import os
os.makedirs("project0/output", exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
# Also save PDF
fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
plt.close()
print(f"Saved: {output_path}")
print(f"Saved: {output_path.replace('.png', '.pdf')}")
