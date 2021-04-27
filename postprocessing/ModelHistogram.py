import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import PercentFormatter
from ExperimentChooser import get_experiment_parameters


def get_counts():
    models = ['no hist', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
    datasets = ['assistment', 'citizen_science', 'mooc']
    rows = []
    for dataset in datasets:
        params = get_experiment_parameters(dataset, True)
        version, user_type, target_col, model_type, performance_metric, bin_size, min_hist_len_to_test = params
        results_dir = 'C:/Users/Jonma/Documents/BGU/Thesis/results/%s' % model_type
        log_dir = '%s/%s/%s/%s/%s' % (results_dir, dataset, version, user_type, performance_metric)
        df = pd.read_csv('%s/model_comparison_%s.csv' % (log_dir, experiment_subset), index_col='model')
        counts = pd.Series(df['best count'])
        row = [dataset] + [counts[model] for model in models]
        rows.append(row)
    pd.DataFrame(rows, columns=['dataset'] + models).to_csv(
        '%s/model_counts_%s.csv' % (dir_path, experiment_subset), index=False)


experiment_subset = 'valid_bins'
# experiment_subset = 'test_bins_with_best_with_best'

experiment_version = 'fit on train and valid'

dir_path = 'C:/Users/Jonma/Documents/BGU/Thesis/results/%s' % experiment_version
# get_counts()
matplotlib.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()

# color_db1 = 'dimgrey'
# color_db2 = 'darkgrey'
# color_db3 = 'lightgrey'

color_db1 = 'mediumorchid'
color_db2 = 'mediumseagreen'
color_db3 = 'orange'

dataset_real_names = {'assistment': 'ASSISTments', 'mooc': 'MOOC', 'citizen_science': 'GZ'}
df = pd.read_csv('%s/model_counts_%s.csv' % (dir_path, experiment_subset))
datasets = df['dataset']
x = np.arange(9)
row_1 = df.loc[0][1:]
row_2 = df.loc[1][1:]
row_3 = df.loc[2][1:]
y_1 = 100 * row_1/np.sum(row_1)
y_2 = 100 * row_2/np.sum(row_2)
y_3 = 100 * row_3/np.sum(row_3)
ax.bar(x - 0.2, y_1, width=0.2, label=dataset_real_names[datasets[0]],  color=color_db1)
ax.bar(x, y_2, width=0.2, label=dataset_real_names[datasets[1]], color=color_db2)
ax.bar(x + 0.2, y_3, width=0.2, label=dataset_real_names[datasets[2]], color=color_db3)
plt.xticks(x, ['m%d' % i for i in range(9)])
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
plt.ylabel('percent of times chosen')
ax.legend()
plt.savefig('%s/model_counts_%s.png' % (dir_path, experiment_subset), bbox_inches='tight')
plt.show()
