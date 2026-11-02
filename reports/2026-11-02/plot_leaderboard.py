"""Reproduce the public README figures and table from the adjacent aggregate JSON.

Run with matplotlib installed: python reports/2026-11-02/plot_leaderboard.py
No benchmark inputs, credentials, or network access are needed.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'leaderboard.json').read_text())['public_leaderboard']
ROWS = sorted(DATA['ranking']['leaderboard'], key=lambda x: x['rank'])
NEW = {x['model']: x['label'] for x in DATA['generation_coverage']}
LABELS = {
    'opus-5-high': 'Claude Opus 5 · high', 'opus-4.8-xhigh': 'Claude Opus 4.8 · xhigh',
    'opus-4.8-off': 'Claude Opus 4.8 · off', 'sonnet-4.6-low': 'Claude Sonnet 4.6 · low',
    'gpt-5.5-high': 'GPT 5.5 · high', 'gpt-5.5-pro-med': 'GPT 5.5 Pro · medium',
    'gpt-5.4-mini-none': 'GPT 5.4 Mini · none', 'kimi-k2.6': 'Kimi K2.6',
    'gemini-3.1-pro-high': 'Gemini 3.1 Pro · high', 'gemini-3.5-flash-low': 'Gemini 3.5 Flash · low',
    'gemini-3.7-flash-medium': 'Gemini 3.7 Flash · medium', 'gemma-4-31b-high': 'Gemma 4 31B · high',
    **NEW,
}
CRITERIA = {
    'q01': 'Concept\nrelevance', 'q02': 'Grouping', 'q03': 'Priority',
    'q04': 'Time\nallocation', 'q05': 'Practice\nalignment', 'q06': 'Difficulty',
    'q07': 'Answer\naccuracy', 'q08': 'Strategies /\nformats',
}
assert len(ROWS) == 17 and set(CRITERIA) == set(DATA['per_rubric'])
assert all(DATA['rank_intervals'][r['model']]['n_valid'] == 500 for r in ROWS)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})

# Point estimates and student-cluster percentile ranges.
fig, ax = plt.subplots(figsize=(11.6, 8.5), layout='constrained')
for i, row in enumerate(ROWS):
    name, rank = row['model'], row['rank']
    limits = DATA['rank_intervals'][name]
    color = '#0072B2' if name in NEW else '#555E66'
    ax.plot([limits['low'], limits['high']], [i, i], color=color, lw=2.1, zorder=2)
    ax.scatter([rank], [i], marker='D' if name in NEW else 'o', s=49,
               color=color, edgecolor='white', linewidth=.6, zorder=3)
ax.set_yticks(range(len(ROWS)), [LABELS[r['model']] for r in ROWS], fontsize=11)
ax.invert_yaxis()
ax.set_xticks(range(1, 18))
ax.set_xlim(.4, 17.6)
ax.set_xlabel('Overall rank  ·  1 = highest estimated quality', labelpad=12)
ax.xaxis.grid(True, color='#E0E5E8', lw=.8)
ax.tick_params(axis='y', length=0)
ax.set_title('Overall AI-judged ranking', loc='left', fontsize=19, weight='bold', pad=18)
ax.legend(handles=[Line2D([], [], marker='o', color='#555E66', linestyle='None', label='Historical baseline'),
                   Line2D([], [], marker='D', color='#0072B2', linestyle='None', label='New generator')],
          loc='upper right', frameon=False, fontsize=10)
fig.savefig(ROOT / 'figure-overall-ranks.png', dpi=180)
plt.close(fig)

# Independent criterion fits, not filtered overall ranks.
criteria_ranks = {criterion: {r['model']: r['rank'] for r in DATA['per_rubric'][criterion]['leaderboard']}
                  for criterion in CRITERIA}
matrix = np.array([[criteria_ranks[c][row['model']] for c in CRITERIA] for row in ROWS])
fig, ax = plt.subplots(figsize=(12.2, 9), layout='constrained')
im = ax.imshow(matrix, cmap='Blues_r', vmin=1, vmax=17, aspect='auto')
ax.set_xticks(range(8), list(CRITERIA.values()), fontsize=10)
ax.set_yticks(range(len(ROWS)), [LABELS[r['model']] + ('  ◆' if r['model'] in NEW else '') for r in ROWS], fontsize=10.5)
ax.tick_params(length=0, pad=6)
ax.xaxis.tick_top()
for y in range(len(ROWS)):
    for x in range(len(CRITERIA)):
        value = matrix[y, x]
        ax.text(x, y, f'{value:g}', ha='center', va='center',
                color='white' if value <= 7 else '#172B35', fontsize=10, weight='bold')
ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
ax.set_yticks(np.arange(-.5, len(ROWS), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.5)
ax.tick_params(which='minor', bottom=False, left=False)
ax.set_title('Rank by quality criterion', loc='left', fontsize=19, weight='bold', pad=26)
fig.colorbar(im, ax=ax, shrink=.65, pad=.025, label='Rank  ·  1 = highest estimated quality')
fig.savefig(ROOT / 'figure-criterion-ranks.png', dpi=180)
plt.close(fig)

print('| Rank | Generator system | 95% rank range | BT score | Comparisons | Students |')
print('| ---: | --- | ---: | ---: | ---: | ---: |')
for row in ROWS:
    limits = DATA['rank_intervals'][row['model']]
    print(f"| {row['rank']:.0f} | {LABELS[row['model']]}{' *' if row['model'] in NEW else ''} | "
          f"{limits['low']:.0f}–{limits['high']:.0f} | {row['score']:.3f} | "
          f"{row['n_comparisons']} | {row['n_subjects']} |")
