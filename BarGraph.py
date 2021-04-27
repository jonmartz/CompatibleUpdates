import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

title = 'ASSISTments Dataset'
experiment_version = 'fit on train and valid'
normalize_to_oracle = False
plot_std = False
col_for_overlay = 0
matplotlib.rcParams.update({'font.size': 14})
overlay_fontsize = 13

# plt.ylim(0.7, 0.85)

# models = {
#     'oracle': 'tab:orange',
#     # 'meta-oracle': 'tab:pink',
#     # 'meta': 'tab:purple',
#     'best-valid': 'tab:red',
#     'baseline': 'tab:blue',
# }
models = {
    'oracle': 'dimgray',
    # 'meta-oracle': 'tab:pink',
    # 'meta': 'black',
    'best-valid': 'darkgray',
    'baseline': 'lightgray',
}
model_official_names = {
    'oracle': 'oracle',
    'meta': 'meta-learning',
    'best-valid': 'personalized',
    'baseline': 'baseline',
}

# matplotlib.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})
dir_path = 'C:/Users/Jonma/Documents/BGU/Thesis/results/%s' % experiment_version
df = pd.read_csv('%s/results.csv' % dir_path)
x = np.arange(len(df))  # the x locations for the groups
# width = 0.35       # the width of the bars: can also be len(x) sequence

ys = [df[f'{m} avg'] for m in models]
if normalize_to_oracle:
    ys = [i / df['oracle avg'] for i in ys]

# if plot_std:
#     std_oracle = df['oracle std'] / df['oracle avg']
#     std_personalized = df['best-valid std'] / df['oracle avg']
#     std_baseline = df['baseline std'] / df['oracle avg']
#     plt.bar(x, y_oracle, label='oracle', color=color_oracle,
#             yerr=std_oracle, error_kw=dict(ecolor='gray'))
#     plt.bar(x + 0.05, y_personalized, label='personalized', color=color_personalized,
#             yerr=std_personalized, error_kw=dict(ecolor='gray'))
#     plt.bar(x + 0.1, y_baseline, label='baseline', color=color_baseline,
#             yerr=std_baseline, error_kw=dict(ecolor='gray'))
# else:
for i, item in enumerate(models.items()):
    model, color = item
    plt.bar(x + 0.05 * i, ys[i], label=model, color=color)

plt.ylabel('normalized AUTC' if normalize_to_oracle else 'AUTC')
plt.xticks(x, df['dataset'])
# plt.yticks([0, 1], ['h1', 'oracle'], rotation=90, verticalalignment='center')
# plt.legend(loc='lower right')

# Add overlayed text
for i, model in enumerate(models.keys()):
    min_y = ys[i + 1][col_for_overlay] if i < len(models) - 1 else 0
    plt.text(col_for_overlay + 0.05 * i, min_y + (ys[i][col_for_overlay] - min_y) / 2, model_official_names[model],
             fontsize=overlay_fontsize, horizontalalignment='center', verticalalignment='center', color='white')

# # these are matplotlib.patch.Patch properties
# text_str = 'A = ASSISTments\nG = GZ\nM = MOOC'
# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# plt.text(-0.4, 0.85, text_str, fontsize=12, verticalalignment='top', bbox=props)

plt.axhline(y=0.5, color='black', linestyle='--')

# plt.title('meta-task: multi-label, candidates: all 9 models')
plt.title(title)
plt.savefig('%s/bar_graph.png' % dir_path, bbox_inches='tight')
# plt.savefig('%s/bar_graph.png' % dir_path)

plt.show()
