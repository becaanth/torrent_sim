import os
import pandas as pd

import matplotlib.pyplot as plt

print("Files in working directory:", os.listdir("."))
run = "multi"
path = f"morris/out/{run}/morris_comparison.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    print("\nColumns:", df.columns.tolist())
    print("\nHead:\n", df.head())
    print("\nInfo:")
    print(df.info())

print("Strategies:", df['strategy'].unique())
print("Outputs:", df['output'].unique())
print("Parameters:", df['parameter'].unique())

# Let's inspect data grouped by output
outputs = df['output'].unique()
strategies = df['strategy'].unique()

print(outputs)
print(strategies)

fig, axes = plt.subplots(1, len(outputs), figsize=(20, 8))

if len(outputs) == 1:
    axes = [axes]

colors = {'rarest_random': 'tab:orange', 'sequential': 'tab:green', 'cascading': 'tab:blue'}

for ax, out in zip(axes, outputs):
    sub = df[df['output'] == out]
    for strat in strategies:
        strat_df = sub[sub['strategy'] == strat]
        ax.scatter(strat_df['mu_star'], strat_df['sigma'], label=strat, color=colors.get(strat, 'tab:gray'), s=80)
        for _, row in strat_df.iterrows():
            ax.annotate(row['parameter'], (row['mu_star'], row['sigma']), fontsize=10, xytext=(4, 4), textcoords='offset points', color=colors.get(strat, 'tab:gray'))
    ax.set_title(f"Output: {out}")
    ax.set_xlabel("mu_star (Mean absolute elementary effect)")
    ax.set_ylabel("sigma (Standard deviation / Interaction effect)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect(1.0)
    ax.legend(title="Strategy")

plt.tight_layout()
plt.savefig("morris_scatter.png", dpi=500)
print("Saved morris_scatter.png")