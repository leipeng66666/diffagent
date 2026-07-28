#!/usr/bin/env python3
"""
Generate system flowchart for the Table Data AI Agent
Creates a publication-ready diagram showing the system architecture and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up figure with white background
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
color_input = '#E8F4F8'
color_process = '#B3E5FC'
color_analysis = '#81D4FA'
color_output = '#4FC3F7'
color_decision = '#FFE082'
color_storage = '#F8BBD0'

def draw_box(ax, x, y, width, height, text, color, fontsize=10, fontweight='normal'):
    """Draw a rounded rectangle box with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         edgecolor='#333', facecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight=fontweight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=20, 
                           linewidth=2, color='#333')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=9, style='italic', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
ax.text(5, 11.5, 'Table Data AI Agent System Flowchart', 
        ha='center', fontsize=16, fontweight='bold')

# Layer 1: Input
draw_box(ax, 5, 10.5, 2, 0.6, 'User Query / Data Upload', color_input, fontsize=11, fontweight='bold')

# Layer 2: Data Loading & Processing
draw_arrow(ax, 5, 10.2, 5, 9.7)
draw_box(ax, 2.5, 9.2, 2, 0.8, 'Data Loader\n(CSV/Excel/JSON)', color_process)
draw_box(ax, 5, 9.2, 2, 0.8, 'Encoding Detection\n(UTF-8/Latin-1/CP1252)', color_process)
draw_box(ax, 7.5, 9.2, 2, 0.8, 'Data Validation\n& Normalization', color_process)

draw_arrow(ax, 5, 10.2, 2.5, 9.6)
draw_arrow(ax, 5, 10.2, 5, 9.6)
draw_arrow(ax, 5, 10.2, 7.5, 9.6)

# Layer 3: Query Analysis
draw_arrow(ax, 5, 8.8, 5, 8.3)
draw_box(ax, 5, 7.8, 3, 0.8, 'Semantic Parser\n(Intent Detection)', color_analysis)

# Layer 4: Column Mapping
draw_arrow(ax, 5, 7.4, 5, 6.9)
draw_box(ax, 5, 6.4, 3, 0.8, 'Intelligent Column Mapper\n(LLM-based)', color_analysis)

# Layer 5: Data Filtering & Pairing
draw_arrow(ax, 5, 6.0, 5, 5.5)
draw_box(ax, 2.5, 5, 2.2, 0.8, 'Material Filter\n(CH4/CO2)', color_process)
draw_box(ax, 5, 5, 2.2, 0.8, 'Temperature Pairing\n(Similar Temps)', color_process)
draw_box(ax, 7.5, 5, 2.2, 0.8, 'Unit Validation\n(m²/s variants)', color_process)

draw_arrow(ax, 5, 5.5, 2.5, 5.4)
draw_arrow(ax, 5, 5.5, 5, 5.4)
draw_arrow(ax, 5, 5.5, 7.5, 5.4)

# Layer 6: Separation Analysis
draw_arrow(ax, 5, 4.6, 5, 4.1)
draw_box(ax, 5, 3.6, 3.5, 0.8, 'Calculate Log10(D_max/D_min)\nfor Each Zeolite Pair', color_analysis, fontweight='bold')

# Layer 7: Ranking & Table Generation
draw_arrow(ax, 5, 3.2, 5, 2.7)
draw_box(ax, 5, 2.2, 3.5, 0.8, 'Rank Zeolites by Separation\nPerformance (Log10 Scale)', color_analysis, fontweight='bold')

# Layer 8: LLM Analysis
draw_arrow(ax, 5, 1.8, 5, 1.3)
draw_box(ax, 5, 0.8, 3.5, 0.8, 'LLM Analysis\n(English Response)', color_output, fontweight='bold')

# Layer 9: Output
draw_arrow(ax, 5, 0.4, 5, -0.1)
draw_box(ax, 5, -0.6, 2.5, 0.6, 'Final Answer\n(Best Zeolite & Evidence)', color_output, fontsize=11, fontweight='bold')

# Add legend
legend_y = 11
ax.text(0.3, legend_y, 'Legend:', fontsize=10, fontweight='bold')
legend_items = [
    (color_input, 'Input'),
    (color_process, 'Processing'),
    (color_analysis, 'Analysis'),
    (color_output, 'Output')
]
for i, (color, label) in enumerate(legend_items):
    y = legend_y - 0.5 - (i * 0.4)
    rect = mpatches.Rectangle((0.3, y - 0.15), 0.3, 0.3, 
                              facecolor=color, edgecolor='#333', linewidth=1)
    ax.add_patch(rect)
    ax.text(0.8, y, label, fontsize=9, va='center')

# Add key features on the right
features_x = 9.2
features_y = 10
ax.text(features_x, features_y, 'Key Features:', fontsize=10, fontweight='bold', ha='right')
features = [
    '✓ Multi-encoding support',
    '✓ LLM-based column mapping',
    '✓ Temperature-based pairing',
    '✓ Log10 separation metric',
    '✓ English output',
    '✓ Zeolite ranking'
]
for i, feature in enumerate(features):
    ax.text(features_x, features_y - 0.5 - (i * 0.4), feature, fontsize=8, ha='right',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.7, pad=0.3))

plt.tight_layout()
plt.savefig('system_flowchart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Flowchart saved as: system_flowchart.png")
plt.close()

# Also create a simplified version
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'System Architecture Overview', 
        ha='center', fontsize=14, fontweight='bold')

# Simplified flow
stages = [
    (5, 8.5, 'Data Input\n(CSV/Excel)', color_input),
    (5, 7.5, 'Data Processing\n& Validation', color_process),
    (5, 6.5, 'Query Analysis\n& Mapping', color_analysis),
    (5, 5.5, 'Material Filtering\n& Pairing', color_process),
    (5, 4.5, 'Separation Analysis\n(Log10 Metric)', color_analysis),
    (5, 3.5, 'Zeolite Ranking\n& Table Generation', color_analysis),
    (5, 2.5, 'LLM Response\nGeneration', color_output),
    (5, 1.5, 'Final Answer\n(English)', color_output),
]

for i, (x, y, text, color) in enumerate(stages):
    draw_box(ax, x, y, 2.5, 0.7, text, color, fontsize=10)
    if i < len(stages) - 1:
        draw_arrow(ax, x, y - 0.35, x, stages[i+1][1] + 0.35)

# Add side annotations
annotations = [
    (7.5, 8.5, 'Multi-format\nencoding support'),
    (7.5, 7.5, 'Normalize units\n& values'),
    (7.5, 6.5, 'LLM-based\ncolumn detection'),
    (7.5, 5.5, 'Temperature\nmatching'),
    (7.5, 4.5, 'Log10(D_max/D_min)\nfor each pair'),
    (7.5, 3.5, 'Rank by\nseparation power'),
    (7.5, 2.5, 'English system\nprompt'),
    (7.5, 1.5, 'Best zeolite\n+ evidence'),
]

for x, y, text in annotations:
    ax.text(x, y, text, fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.8, pad=0.4))
    ax.plot([6.3, 7.2], [y, y], 'k-', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('system_architecture_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Simplified architecture saved as: system_architecture_simple.png")
plt.close()

print("\nFlowcharts generated successfully!")
print("Files created:")
print("  1. system_flowchart.png - Detailed flowchart with all components")
print("  2. system_architecture_simple.png - Simplified architecture overview")
