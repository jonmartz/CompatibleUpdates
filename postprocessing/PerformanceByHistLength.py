import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

dir = 'C:/Users/Jonma/Documents/BGU/Thesis/results/large experiments/citizen_science/exploration/CART autotuned 70 train/user_id/auc'
cmap_baseline = plt.cm.get_cmap('Greys')
cmap_m1 = plt.cm.get_cmap('Blues')
color_min = 0.3
color_max = 1.0
bin_size = 10

df = pd.read_csv('%s/test_log.csv' % dir)
groups_by_weight = df.groupby('weight')
first = True
for weight, df_weight in groups_by_weight:
    x = np.mean(df_weight['len'].to_numpy().reshape(-1, bin_size), axis=1)
    if first:  # plot h1
        first = False
        y = np.mean(df_weight['h1_acc'].to_numpy().reshape(-1, bin_size), axis=1)
        plt.plot(x, y, label='h1', marker='s', markersize=4, color='red')

    color = color_min + (color_max - color_min) * weight
    y = np.mean(df_weight['no hist y'].to_numpy().reshape(-1, bin_size), axis=1)
    plt.plot(x, y, label='baseline w=%s' % weight, marker='.', color=cmap_baseline(color))
    # y = np.mean(df_weight['L1 y'].to_numpy().reshape(-1, bin_size), axis=1)
    # plt.plot(x, y, label='m1 w=%s' % weight, marker='.', color=cmap_m1(color))

plt.xlabel('history length')
plt.ylabel('AUC')
plt.legend()
plt.savefig('performance_by_hist_length.png', bbox_inches='tight')
plt.title('GZ experiment')
plt.show()
