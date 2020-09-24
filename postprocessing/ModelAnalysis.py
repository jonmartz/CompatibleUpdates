import pandas as pd
from joblib import load
from sklearn import tree
import matplotlib.pyplot as plt

# dataset = 'assistment'
# # model_name = 'user_85828 fold_1 inner_fold_0 model_no hist diss_w_0.60 com_0.787 acc_0.609'
# # fig, ax = plt.subplots(figsize=(9, 4))
# # model_name = 'user_85828 fold_1 inner_fold_0 model_no hist diss_w_0.65 com_0.820 acc_0.591'
# # fig, ax = plt.subplots(figsize=(10, 5))
# # model_name = 'user_85828 fold_1 inner_fold_0 model_L1 diss_w_0.20 com_0.918 acc_0.700'
# # fig, ax = plt.subplots(figsize=(9, 5))
# # model_name = 'user_75169 fold_0 inner_fold_0 model_no hist diss_w_0.60 com_0.846 acc_0.643'
# # fig, ax = plt.subplots(figsize=(11, 6))
# # model_name = 'user_75169 fold_0 inner_fold_0 model_L1 diss_w_0.80 com_0.949 acc_0.686'
# # fig, ax = plt.subplots(figsize=(14, 7))
# # META
# model_name = 'meta_model_fold_0'
# fig, ax = plt.subplots(figsize=(9, 5))


# dataset = 'salaries'
# # model_name = 'user_Husband fold_14 inner_fold_0 model_no hist diss_w_0.60 com_0.857 acc_0.630'
# # fig, ax = plt.subplots(figsize=(9.5, 5))
# model_name = 'user_Husband fold_14 inner_fold_0 model_L1 diss_w_0.60 com_0.911 acc_0.655'
# fig, ax = plt.subplots(figsize=(9.5, 5))

# dataset = 'recividism'
# # model_name = 'user_African-American fold_5 inner_fold_0 model_no hist diss_w_0.40 com_0.886 acc_0.639'
# # fig, ax = plt.subplots(figsize=(15, 4))
# # model_name = 'user_African-American fold_5 inner_fold_0 model_L4 diss_w_0.80 com_0.922 acc_0.653'
# # fig, ax = plt.subplots(figsize=(17, 5))
# # META
# model_name = 'meta_model_fold_2'
# fig, ax = plt.subplots(figsize=(28, 11))

dataset = 'citizen_science'
# model_name = 'user_472 fold_3 inner_fold_0 model_no hist diss_w_0.50 com_0.711 acc_0.744'
# fig, ax = plt.subplots(figsize=(10, 4))
model_name = 'user_472 fold_3 inner_fold_0 model_L1 diss_w_0.40 com_0.868 acc_0.878'
fig, ax = plt.subplots(figsize=(18, 7))

# dataset = 'mooc'
# # model_name = 'user_10067 fold_5 inner_fold_0 model_no hist diss_w_0.30 com_0.765 acc_0.714'
# # fig, ax = plt.subplots(figsize=(7, 6))
# # model_name = 'user_10067 fold_5 inner_fold_0 model_L1 diss_w_0.30 com_0.941 acc_0.857'
# # fig, ax = plt.subplots(figsize=(19, 7))
# # model_name = 'user_10067 fold_0 inner_fold_0 model_L2 diss_w_0.10 com_0.944 acc_0.786'
# # fig, ax = plt.subplots(figsize=(12, 7))
# # model_name = 'user_10067 fold_3 inner_fold_0 model_L1 diss_w_0.07 com_0.842 acc_0.821'
# # fig, ax = plt.subplots(figsize=(20, 8))
# model_name = 'user_10067 fold_0 inner_fold_0 model_L3 diss_w_0.03 com_0.833 acc_0.857'
# fig, ax = plt.subplots(figsize=(15, 7))

do_meta_agent = False
# do_meta_agent = True

# for fold in range(6):
#     model_name = 'meta_model_fold_%d' % fold
# print('\nfold %d' % fold)

if do_meta_agent:
    cols = pd.read_csv('%s/meta_feature_names.csv' % dataset).columns
    feature_names = cols[:-3]
    class_names = cols[-3:]
    file_name = '%s/meta_models/%s.joblib' % (dataset, model_name)
    model = load(file_name)
    title = model_name
else:
    cols = pd.read_csv('%s/feature_names.csv' % dataset).columns
    feature_names = cols[:-1]
    target_class = cols[-1]
    class_names = ['%s=%d' % (target_class, i) for i in range(2)]
    file_name = '%s/%s.joblib' % (dataset, model_name)
    model = load(file_name).predictor
    title = model_name[:-32] + '\n' + model_name[-31:]
# model
annotations = tree.plot_tree(model, feature_names=feature_names, class_names=class_names, fontsize=10,
                             filled=True, impurity=False, proportion=True, rounded=True)
plt.title(title)
plt.savefig('%s/%s.png' % (dataset, model_name), bbox_inches='tight')
plt.show()

print('features used:')
features_used = set()
for annotation in annotations:
    text = annotation._text
    if ' <= ' in text:
        features_used.add(text.split(' <= ')[0])
print(features_used)
