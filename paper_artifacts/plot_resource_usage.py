import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# --- 1. Your Data ---

# Include max_distortions = 20
x = [1, 2, 3, 4, 5, 10, 15, 20]

# --- Time Data ---
x = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
embed_time_1 = [9.02, 8.72, 10.25, 11.42, 12.00, 15.23, 18.21, 22.93, 19.09, 21.85]
embed_time_20 = [52.98, 54.14, 55.86, 56.87, 59.00, 65.23, 71.55, 79.85, 80.67, 85.91]
nli_time      = [19.24, 54.39, 105.93, 176.86, 267.76, 999.52, 2272.39, 4231.36, 5864.47, 9205.72]


# --- GPU Data (NLI) ---
nli_gpu_delta = [1848.2, 1945.7, 1945.7, 2073.1, 2073.3, 4396.5, 6356.8, 6356.8, 6444.06, 6444.78]
nli_peak      = [3541.6, 3647.2, 3647.2, 3774.9, 3774.9, 6185.4, 8145.6, 8145.6, 8145.6, 8145.6]

# --- 2. Reshape Data for Seaborn (Long-form) ---

# Create the "Time" DataFrame
df_time = pd.DataFrame({
    'Max Distortions': np.concatenate([x, x, x]),
    'Time (s)': np.concatenate([embed_time_1, embed_time_20, nli_time]),
    'Metric': (['Embed Time (1 trial)'] * len(x) +
               ['Embed Time (20 trials)'] * len(x) +
               ['NLI Time'] * len(x))
})

# Create the "GPU" DataFrame
df_gpu = pd.DataFrame({
    'Max Distortions': np.concatenate([x, x]),
    'GPU Memory (MB)': np.concatenate([nli_gpu_delta, nli_peak]),
    'Metric': (['NLI $\Delta$GPU'] * len(x) +
               ['NLI Peak GPU'] * len(x))
})

# --- 3. Create the Plot ---

# Set the Seaborn style
sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.5)

# Side-by-side panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=False)

# --- Panel (a): Processing Time (Left) ---
sns.lineplot(
    data=df_time,
    x='Max Distortions',
    y='Time (s)',
    hue='Metric',
    style='Metric',
    markers=True,
    markersize=8,
    lw=2.5,
    ax=ax1
)

ax1.set_yscale('log')
ax1.set_ylabel('Time (s) - Logarithmic Scale')
ax1.set_xlabel('Sampling scale')
ax1.set_xticks([1, 2, 3, 4, 5, 10, 15, 20, 25, 30])
ax1.text(0.5, -0.18, '(a)', transform=ax1.transAxes,
         fontsize=16, fontweight='bold', ha='center', va='top')
sns.move_legend(ax1, "upper left", bbox_to_anchor=(0, 1))

# Add data labels
for line_data in [embed_time_1, embed_time_20, nli_time]:
    for i, val in enumerate(line_data):
        ax1.text(x[i], val, f'{val:.0f}', ha='center', va='bottom', fontsize=9)

# --- Panel (b): GPU Memory (Right) ---
sns.lineplot(
    data=df_gpu,
    x='Max Distortions',
    y='GPU Memory (MB)',
    hue='Metric',
    style='Metric',
    markers=True,
    markersize=8,
    lw=2.5,
    ax=ax2
)

ax2.set_ylim(bottom=0)
ax2.set_ylabel('GPU Memory (MB)')
ax2.set_xlabel('Sampling scale - Logarithmic Scale')
ax2.text(0.5, -0.18, '(b)', transform=ax2.transAxes,
         fontsize=16, fontweight='bold', ha='center', va='top')
sns.move_legend(ax2, "upper left", bbox_to_anchor=(0, 1))

# Add data labels
for line_data in [nli_gpu_delta, nli_peak]:
    for i, val in enumerate(line_data):
        ax2.text(x[i], val, f'{val:.0f}', ha='center', va='bottom', fontsize=9)
ax2.set_xscale('log')
ax2.set_xticks([1, 2, 3, 4, 5, 10, 15, 20, 25, 30])
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# --- 4. Final Touches ---
plt.tight_layout(pad=1.0)
plt.savefig('seaborn_side_by_side.pdf', bbox_inches='tight')
plt.show()