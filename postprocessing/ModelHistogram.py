import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import PercentFormatter


def get_counts():
    models = ['no hist', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
    datasets = ['assistment', 'mooc', 'citizen_science']
    rows = []
    for dataset in datasets:
        df = pd.read_csv('%s/best_models_valid_bins.csv' % dataset)
        counts = df['model'].value_counts()
        row = [dataset]
        keys = counts.keys()
        for model in models:
            if model in keys:
                row.append(counts[model])
            else:
                row.append(0)
        rows.append(row)
    pd.DataFrame(rows, columns=['dataset'] + models).to_csv('model_counts.csv', index=False)


# get_counts()

matplotlib.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()

df = pd.read_csv('model_counts.csv')
datasets = df['dataset']
x = np.arange(9)
row_1 = df.loc[0][1:]
row_2 = df.loc[1][1:]
row_3 = df.loc[2][1:]
y_1 = 100 * row_1/np.sum(row_1)
y_2 = 100 * row_2/np.sum(row_2)
y_3 = 100 * row_3/np.sum(row_3)
ax.bar(x - 0.2, y_1, width=0.2, label=datasets[0],  color='dimgrey')
ax.bar(x, y_2, width=0.2, label=datasets[1], color='darkgrey')
ax.bar(x + 0.2, y_3, width=0.2, label=datasets[2], color='lightgrey')
plt.xticks(x, df.columns[1:])
ax.yaxis.set_major_formatter(PercentFormatter())
plt.ylabel('percent of times chosen')
ax.legend()
plt.savefig('model_counts.png', bbox_inches='tight')
plt.show()