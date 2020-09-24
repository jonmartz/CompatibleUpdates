import pandas as pd
import os
import matplotlib.pyplot as plt

dataset = 'assistment'
# dataset = 'recividism'

ver = 'meta-learning'
meta_ver = 2
weighted = True
extended = False
percentage = False

# load stuff
if dataset == 'assistment':
    user_col = 'user_id'
elif dataset == 'recividism':
    user_col = 'race'
log_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/results/large experiments/%s/%s/%s/acc' % (dataset, ver, user_col)
df_path = '%s/meta_results_ver_%d' % (log_dir, meta_ver)
if weighted:
    df_path += ' weighted'
if extended:
    df_path += '_extended'
df_path += '.csv'
df = pd.read_csv(df_path)
out_dir = '%s/sensitivity_analysis' % log_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
agents = ['best_train', 'best_valid', 'meta_agent', 'golden_standard']
meta_features_names = list(pd.read_csv('%s/meta_feature_names.csv' % dataset).columns[:-3])
baseline_autc = df['baseline_autc']
# make plots
for meta_feature_name in meta_features_names:
    print(meta_feature_name)
    meta_feature = df[meta_feature_name]
    scale = 20
    markersize = scale * len(agents)
    for agent in agents:
        autc = df['%s_autc' % agent]
        if percentage:
            autc = 100 * ((autc / baseline_autc) - 1)
        else:
            autc = autc - baseline_autc
        plt.scatter(meta_feature, autc, label=agent, s=markersize)
        markersize -= scale
    plt.grid()
    plt.legend()
    plt.xlabel(meta_feature_name)
    if percentage:
        plt.ylabel('+AUTC% vs baseline')
    else:
        plt.ylabel('AUTC(m) - AUTC(baseline)')
    plt.title('%s sensitivity analysis' % meta_feature_name)
    plt.savefig('%s/%s.png' % (out_dir, meta_feature_name), bbox_inches='tight')
    plt.clf()
