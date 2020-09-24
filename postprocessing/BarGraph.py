import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 15})
# matplotlib.rcParams.update({'font.size': 16, 'font.family': 'Times New Roman'})

fontsize = 14
df = pd.read_csv('results.csv')
# N = 5
# oracle_avg = (20, 35, 30, 35, 27)
# womenMeans = (25, 32, 34, 20, 25)
# menStd = (2, 3, 4, 1, 2)
# womenStd = (3, 5, 2, 3, 3)
x = np.arange(len(df))    # the x locations for the groups
# width = 0.35       # the width of the bars: can also be len(x) sequence

# plt.bar(x, df['oracle avg']/df['oracle avg'], yerr=df['oracle std']/df['oracle avg'], label='oracle')
y_oracle = df['oracle avg']/df['oracle avg']
y_personalized = df['best-valid avg'] / df['oracle avg']
y_baseline = df['baseline avg']/df['oracle avg']

std_oracle = df['oracle std']/df['oracle avg']
std_personalized = df['best-valid std'] / df['oracle avg']
std_baseline = df['baseline std']/df['oracle avg']

plt.bar(x, y_oracle, label='oracle', color='dimgrey',
        yerr=std_oracle, error_kw=dict(ecolor='gray'))
plt.bar(x + 0.05, y_personalized, label='personalized', color='darkgrey',
        yerr=std_personalized, error_kw=dict(ecolor='gray'))
plt.bar(x+0.1, y_baseline, label='baseline', color='lightgrey',
        yerr=std_baseline, error_kw=dict(ecolor='gray'))
# p2 = plt.bar(x, womenMeans, width,
#              bottom=oracle_avg, yerr=womenStd)

plt.ylabel('normalized AUTC')
# plt.title('Scores by group and gender')
plt.xticks(x, df['dataset'])
# plt.yticks(np.arange(0, 81, 10))
# plt.legend(loc='lower right')
plt.text(0, y_personalized[0] + (y_oracle[0] - y_personalized[0]) / 2, 'oracle', fontsize=fontsize,
         horizontalalignment='center', verticalalignment='center')
plt.text(0.05, y_baseline[0] + (y_personalized[0] - y_baseline[0]) / 2, 'personalized', fontsize=fontsize,
         horizontalalignment='center', verticalalignment='center')
plt.text(0.1, y_baseline[0] / 2, 'baseline', fontsize=fontsize,
         horizontalalignment='center', verticalalignment='center')
plt.savefig('bar_graph.png', bbox_inches='tight')

plt.show()