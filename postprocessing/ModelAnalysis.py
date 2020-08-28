import pandas as pd
from joblib import load
from sklearn import tree
import matplotlib.pyplot as plt

# root_dir = 'assistment'
# model_name = 'user_85828 fold_1 inner_fold_0 model_no hist diss_w_0.60 com_0.787 acc_0.609'
# fig, ax = plt.subplots(figsize=(9, 4))
# model_name = 'user_85828 fold_1 inner_fold_0 model_no hist diss_w_0.65 com_0.820 acc_0.591'
# fig, ax = plt.subplots(figsize=(10, 5))
# model_name = 'user_85828 fold_1 inner_fold_0 model_L1 diss_w_0.20 com_0.918 acc_0.700'
# fig, ax = plt.subplots(figsize=(9, 5))
# model_name = 'user_75169 fold_0 inner_fold_0 model_no hist diss_w_0.60 com_0.846 acc_0.643'
# fig, ax = plt.subplots(figsize=(11, 6))
# model_name = 'user_75169 fold_0 inner_fold_0 model_L1 diss_w_0.80 com_0.949 acc_0.686'
# fig, ax = plt.subplots(figsize=(14, 7))

# root_dir = 'salaries'
# # model_name = 'user_Husband fold_14 inner_fold_0 model_no hist diss_w_0.60 com_0.857 acc_0.630'
# # fig, ax = plt.subplots(figsize=(9.5, 5))
# model_name = 'user_Husband fold_14 inner_fold_0 model_L1 diss_w_0.60 com_0.911 acc_0.655'
# fig, ax = plt.subplots(figsize=(9.5, 5))

root_dir = 'recividism'
# model_name = 'user_African-American fold_5 inner_fold_0 model_no hist diss_w_0.40 com_0.886 acc_0.639'
# fig, ax = plt.subplots(figsize=(15, 4))
model_name = 'user_African-American fold_5 inner_fold_0 model_L4 diss_w_0.80 com_0.922 acc_0.653'
fig, ax = plt.subplots(figsize=(17, 5))

feature_names = pd.read_csv('%s/feature_names.csv' % root_dir).columns[:-1]
file_name = '%s/%s.joblib' % (root_dir, model_name)
model = load(file_name)
annotations = tree.plot_tree(model.predictor, feature_names=feature_names, fontsize=10, filled=True, impurity=False,
                             proportion=True, rounded=True)
title = model_name[:-32] + '\n' + model_name[-31:]
plt.title(title)
plt.savefig('%s/%s.png' % (root_dir, model_name), bbox_inches='tight')
plt.show()

print('\nFEATURES USED:')
for annotation in annotations:
    text = annotation._text
    if ' <= ' in text:
        print(text.split(' <= ')[0])