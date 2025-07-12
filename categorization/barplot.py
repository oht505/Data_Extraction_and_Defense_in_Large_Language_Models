import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set CVPR style
mpl.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.linewidth': 1.0,
    'axes.labelsize': 17,
    'axes.titlesize': 11,
    'xtick.labelsize': 15,
    'ytick.labelsize': 11,
    'legend.fontsize': 15,
    'legend.frameon': False,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'savefig.dpi': 600,
    'figure.dpi': 600
})

# Categories & Values
categories = ['Named Indiv.', 'Enron & PII data']
models = ['GPT-2 Nano', 'GPT-2 Large']
dp_status = ['Non-DP', 'DP']

values = {
    'Named Indiv.': {
        'GPT-2 Nano': [407, 194],
        'GPT-2 Large': [54, 21]
    },
    'Enron & PII data': {
        'GPT-2 Nano': [165, 10],
        'GPT-2 Large': [380, 6]
    }
}

color_map = {
    ('GPT-2 Nano', 'Non-DP'): '#1f77b4',     # blue
    ('GPT-2 Nano', 'DP'): '#aec7e8',         # light blue
    ('GPT-2 Large', 'Non-DP'): '#ff7f0e',     # orange
    ('GPT-2 Large', 'DP'): '#ffbb78'          # light orange
}

bar_width = 0.18
model_gap = bar_width * 2.6
category_gap = model_gap * 0.4

positions = []
heights = []
colors = []

start = 0
xticks = []
xtick_labels = []

for category in categories:
    for model in models:
        for i, dp in enumerate(dp_status):
            pos = start + i * bar_width
            positions.append(pos)
            heights.append(values[category][model][i])
            colors.append(color_map[(model, dp)])
        start += model_gap
    xticks.append((positions[-1] + positions[-4]) / 2)
    xtick_labels.append(category)
    start += category_gap

# Plot
plt.figure(figsize=(7, 4))  
plt.bar(positions, heights, width=bar_width, color=colors)

plt.xticks(xticks, xtick_labels)
plt.ylabel('Extracted Samples')

# Arrange the Legend outside
legend_patches = [
    plt.Rectangle((0, 0), 1, 1, color=color_map[('GPT-2 Nano', 'Non-DP')], label='Nano (Non-DP)'),
    plt.Rectangle((0, 0), 1, 1, color=color_map[('GPT-2 Nano', 'DP')], label='Nano (DP)'),
    plt.Rectangle((0, 0), 1, 1, color=color_map[('GPT-2 Large', 'Non-DP')], label='Large (Non-DP)'),
    plt.Rectangle((0, 0), 1, 1, color=color_map[('GPT-2 Large', 'DP')], label='Large (DP)')
]
plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.01, 0.5) )

plt.tight_layout()
plt.savefig("extraction_barplot.pdf", dpi=600, bbox_inches='tight')
plt.show()
