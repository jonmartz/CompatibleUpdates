import csv
import os.path
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
import Models
from sklearn.metrics import confusion_matrix
import seaborn as sn


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


def plot_confusion_matrix(predicted, true, title, path):
    matrix = confusion_matrix(true, predicted)

    true_0_count = matrix[0].sum()
    true_1_count = matrix[1].sum()
    pred_0_count = matrix.transpose()[0].sum()
    pred_1_count = matrix.transpose()[1].sum()

    max_count = max(true_0_count, true_1_count, pred_0_count, pred_1_count)

    df_matrix = pd.DataFrame(matrix, ['true 0', 'true 1'], ['predicted 0', 'predicted 1'])
    sn.set(font_scale=1.5)
    ax = sn.heatmap(df_matrix, annot=True, cbar=False, cmap="YlGnBu", fmt="d",
                    linewidths=.2, linecolor='black', vmin=0, vmax=max_count)

    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)

    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('0.2')

    plt.text(2.05, 0.5, true_0_count, verticalalignment='center')
    plt.text(2.05, 1.5, true_1_count, verticalalignment='center')
    plt.text(0.5, 2.25, pred_0_count, horizontalalignment='center')
    plt.text(1.5, 2.25, pred_1_count, horizontalalignment='center')

    plt.subplots_adjust(top=0.85)
    plt.subplots_adjust(right=0.85)

    plt.title(title)
    plt.savefig(path)
    # plt.show()
    plt.clf()
    sn.set(font_scale=1.0)
    # sn.reset_orig()


# Data-set paths

# dataset_name = 'ednet'
# # data settings
# target_col = 'correct_answer'
# original_categ_cols = ['source', 'platform']
# user_cols = ['user']
# skip_cols = []
# # skip_cols = ['bkt_skill_learn_rate', 'bkt_skill_forget_rate', 'bkt_skill_guess_rate', 'bkt_skill_slip_rate']
# df_max_size = 100000
# # experiment settings
# train_frac = 0.7
# valid_frac = 0.2
# h1_len = 20
# h2_len = 30000
# seeds = range(2)
# inner_seeds = range(2)
# weights_num = 3
# weights_range = [0, 1]
# sim_ann_var = 0.05
# max_sim_ann_iter = -1
# iters_to_cooling = 100
# # model settings
# max_depth = None
# ccp_alphas = [0.009]
# # ccp_alphas = [0.003]
# # ccp_alphas = [i/1000 for i in range(1, 10)] + [i/100 for i in range(1, 10)]
# sample_weights_factor = [0.0, 1.0, 1.0, 1.0]
# best_sample_weight = [0, 0, 0, 0]
# # user settings
# min_hist_len = 0
# max_hist_len = 100000
# current_user_count = 0
# users_to_not_test_on = []
# only_these_users = []
# metric = 'acc'
# # metric = 'auc'

# dataset_name = 'assistment'
# # data settings
# target_col = 'correct'
# original_categ_cols = ['skill', 'tutor_mode', 'answer_type', 'type']
# user_cols = ['user_id']
# # skip_cols = []
# skip_cols = ['skill']
# df_max_size = 100000
# # experiment settings
# train_frac = 0.7
# valid_frac = 0.2
# h1_len = 20
# h2_len = 5000
# seeds = range(50)
# inner_seeds = range(20)
# weights_num = 20
# # seeds = range(2)
# # inner_seeds = range(3)
# # weights_num = 5
# weights_range = [0, 1]
# sim_ann_var = 0.05
# max_sim_ann_iter = -1
# iters_to_cooling = 100
# # model settings
# max_depth = None
# ccp_alphas = [0.005]
# # ccp_alphas = [i / 1000 for i in range(1, 11)]
# sample_weights_factor = [0.0, 1.0, 1.0, 1.0]
# # best_sample_weight = [0.01171477, 0.04833975, 0.699829795, 0.550231695]
# best_sample_weight = [0.0, 0.6352316047435935, 0.3119101971209735, 0.07805665820394585]
# # user settings
# min_hist_len = 300
# max_hist_len = 100000
# current_user_count = 0
# users_to_not_test_on = []
# only_these_users = []

# dataset_name = "mooc"
# # data settings
# target_col = 'Opinion(1/0)'
# original_categ_cols = ['course_display_name', 'post_type', 'CourseType']
# user_cols = ['forum_uid']
# skip_cols = ['up_count', 'reads']
# df_max_size = 100000
# # experiment settings
# train_frac = 0.8
# valid_frac = 0.1
# h1_len = 50
# h2_len = 5000
# seeds = range(1)
# inner_seeds = range(2)
# weights_num = 3
# weights_range = [0, 1]
# sim_ann_var = 0.05
# max_sim_ann_iter = -1
# iters_to_cooling = 100
# # model settings
# max_depth = None
# ccp_alphas = [0.00001]
# # user settings
# min_hist_len = 50
# max_hist_len = 2000
# current_user_count = 0
# users_to_not_test_on = []
# only_these_users = []

dataset_name = "recividism"
# data settings
target_col = 'is_recid'
original_categ_cols = ['sex', 'race', 'age_cat', 'c_charge_degree', 'score_text']
user_cols = ['race']
skip_cols = ['c_charge_desc', 'priors_count']
# skip_cols = ['c_charge_desc', 'age_cat', 'score_text']
# skip_cols = ['c_charge_desc', 'priors_count', 'age_cat', 'score_text']
df_max_size = 100000
# experiment settings
train_frac = 0.8
valid_frac = 0.1
h1_len = 20
h2_len = 5000
seeds = range(2)
inner_seeds = range(2)
weights_num = 5
weights_range = [0, 1]
# sim_ann
sim_ann_var = 0.3
max_sim_ann_iter = -1
iters_to_cooling = 100
# model settings
max_depth = None
ccp_alphas = [0.005]
sample_weights_factor = None
# best_sample_weight = [2.849432394, 0.046259433, 2.915855879, 4.184277544]
# user settings
min_hist_len = 50
max_hist_len = 2000
current_user_count = 0
users_to_not_test_on = []
only_these_users = []
metrics = ['acc']

# dataset_name = "salaries"
# # data settings
# target_col = 'salary'
# original_categ_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                        'native-country']
# # user_cols = ['relationship']
# user_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
# skip_cols = ['fnlgwt', 'education', 'native-country']
# df_max_size = 100000
# # experiment settings
# train_frac = 0.8
# valid_frac = 0.1
# h1_len = 20
# h2_len = 5000
# seeds = range(3)
# inner_seeds = range(5)
# weights_num = 10
# weights_range = [0, 1]
# # sim_ann
# sim_ann_var = 0.05
# max_sim_ann_iter = -1
# iters_to_cooling = 100
# # model settings
# max_depth = None
# ccp_alphas = [0.008]
# # ccp_alphas = [i / 1000 for i in range(1, 11)]
# sample_weights_factor = None
# # sample_weights_factor = [0.0, 1.0, 1.0, 1.0]
# # best_sample_weight = []
# # user settings
# min_hist_len = 50
# max_hist_len = 1000
# current_user_count = 0
# users_to_not_test_on = []
# only_these_users = []

# dataset_name = "hospital_mortality"
# # data settings
# target_col = 'HOSPITAL_EXPIRE_FLAG'
# original_categ_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']
# # user_cols = ['MARITAL_STATUS']
# user_cols = ['ADMISSION_TYPE', 'ETHNICITY']
# skip_cols = []
# df_max_size = 100000
# # experiment settings
# train_frac = 0.8
# h1_len = 50
# h2_len = 5000
# seeds = range(30)
# weights_num = 10
# weights_range = [0, 1]
# # model settings
# max_depth = None
# ccp_alpha = 0.001
# # user settings
# min_hist_len = 50
# max_hist_len = 2000
# current_user_count = 0
# users_to_not_test_on = []
# only_these_users = []

# full_dataset_path = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/mallzee/mallzee.csv'
# results_path = "C:/Users/Jonathan/Documents/BGU/Research/Thesis/results/mallzee"
# target_col = 'userResponse'
# categ_cols = ['Currency', 'TypeOfClothing', 'Gender', 'InStock', 'Brand', 'Colour']
# user_group_names = ['userID']
# skip_cols = []
# df_max_size = -1
# layers = []

# full_dataset_path = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/moviesKaggle/moviesKaggle.csv'
# results_path = "C:/Users/Jonathan/Documents/BGU/Research/Thesis/results/moviesKaggle"
# target_col = 'rating'
# categ_cols = ['original_language']
# user_group_names = ['userId']
# skip_cols = []
# # skip_cols = ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller',
# #              'Horror', 'History', 'Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie']

# dataset = "diabetes"
# target_col = 'Outcome'
# original_categ_cols = ['AgeClass']
# # user_cols = ['relationship', 'race', 'education', 'occupation', 'marital-status', 'workclass', 'sex', 'native-country']
# user_cols = ['AgeClass']
# skip_cols = []
# df_max_size = -1
# layers = [5, 5, 5]
# train_frac = 0.8
# h1_len = 10
# h2_len = 70
# h1_epochs = 1000
# h2_epochs = 500

# dataset = 'abalone'
# target_col = 'Rings'
# original_categ_cols = ['sex']
# user_cols = ['sex']
# skip_cols = []
# df_max_size = -1
# layers = [5]
# train_frac = 0.8
# h1_len = 100
# h2_len = 3000
# h1_epochs = 300
# h2_epochs = 300

# full_dataset_path = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/titanic/titanic.csv'
# results_path = "C:/Users/Jonathan/Documents/BGU/Research/Thesis/results/titanic"
# target_col = 'Survived'
# original_categ_cols = ['Sex', 'Embarked', 'AgeClass']
# user_cols = ['Pclass', 'Sex', 'AgeClass', 'Embarked']
# skip_cols = []
# layers = []
# df_max_size = -1
# train_frac = 0.8
# h1_len = 15
# h2_len = 500
# h1_epochs = 500
# h2_epochs = 800

# dataset = 'breastCancer'
# target_col = 'is_malign'
# original_categ_cols = []
# user_cols = ['uniformity_of_size']
# skip_cols = []
# layers = []
# df_max_size = -1
# train_frac = 0.8
# h1_len = 200
# h2_len = 200
# h1_epochs = 200
# h2_epochs = 200

dataset_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/%s' % dataset_name
result_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/current result'

# experiment settings
chrono_split = False
balance_histories = False

# output settings
make_tradeoff_plots = False
show_tradeoff_plots = False
plot_confusion = False

# model settings
models_to_test = {
    # 'no diss': {'sample_weight': [1, 0, 1, 0], 'color': 'grey'},
    'no hist': {'sample_weight': [1, 1, 0, 0], 'color': 'black'},
    # 'sim_ann': {'sample_weight': best_sample_weight, 'color': 'red'},
    # 'hybrid': {'color': 'green'},
}
parametrized_models = [  # [general_loss, general_diss, hist_loss, hist_diss]
    ['L1', [0, 0, 1, 1]],
    ['L2', [0, 1, 1, 0]],
    ['L3', [0, 1, 1, 1]],
    ['L4', [1, 0, 0, 1]],
    ['L5', [1, 0, 1, 1]],
    ['L6', [1, 1, 0, 1]],
    ['L7', [1, 1, 1, 0]],
    ['L8', [1, 1, 1, 1]],
]
cmap = plt.cm.get_cmap('jet')
for i in range(len(parametrized_models)):
    model = parametrized_models[i]
    models_to_test[model[0]] = {'sample_weight': model[1], 'color': cmap((i + 1) / (len(parametrized_models) + 2))}

# default settings
diss_weights = np.array([i / weights_num for i in range(weights_num + 1)])
diss_weights = (diss_weights * (weights_range[1] - weights_range[0]) + weights_range[0]).tolist()
print('diss_weights = %s' % diss_weights)
model_names = list(models_to_test.keys())

print('steps per model = seeds=%d x inner_seeds=%d x weights=%d = %d'
      % (len(seeds), len(inner_seeds), len(diss_weights), len(seeds) * len(inner_seeds) * len(diss_weights)))

# skip cols
user_cols_not_skipped = []
for user_col in user_cols:
    if user_col not in skip_cols:
        user_cols_not_skipped.append(user_col)
original_categs_not_skipped = []
for categ in original_categ_cols:
    if categ not in skip_cols:
        original_categs_not_skipped.append(categ)
user_cols = user_cols_not_skipped
original_categ_cols = original_categs_not_skipped

dataset_path = '%s/%s.csv' % (dataset_dir, dataset_name)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    with open('%s/parameters.csv' % result_dir, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(['train_frac', 'ccp_alpha', 'dataset_max_size', 'h1_len', 'h2_len', 'seeds', 'weights_num',
                         'weights_range', 'min_hist_len', 'max_hist_len', 'chrono_split', 'balance_histories',
                         'skip_cols'])
        writer.writerow([train_frac, str(ccp_alphas), df_max_size, h1_len, h2_len, len(seeds), weights_num,
                         str(weights_range), min_hist_len, max_hist_len, chrono_split, balance_histories,
                         str(skip_cols)])

# run whole experiment for each user column selection
for user_col in user_cols:
    print('user column = %s' % user_col)
    last_seed_done_by_user = {}

    # create all folders
    result_user_type_dir = '%s/%s' % (result_dir, user_col)
    if not os.path.exists(result_user_type_dir):
        for metric in metrics:
            os.makedirs('%s/%s' % (result_user_type_dir, metric))

        header = ['user', 'len', 'seed', 'inner_seed', 'h1_acc', 'weight']
        for model_name in model_names:
            header.extend(['%s x' % model_name, '%s y' % model_name])
        for metric in metrics:
            for subset_name in ['valid', 'test']:
                with open('%s/%s/%s_log.csv' % (result_user_type_dir, metric, subset_name), 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)

        if make_tradeoff_plots:
            os.makedirs('%s/plots' % result_user_type_dir)

    else:
        df_done = pd.read_csv('%s/%s/valid_log.csv' % (result_user_type_dir, metrics[0]))
        groups_by_user = df_done.groupby('user')
        for user_id, user_group in groups_by_user:
            last_seed = pd.unique(user_group['seed'])[-1]
            last_seed_done_by_user[user_id] = last_seed

    cache_dir = '%s/caches/%s skip_%s max_len_%d min_hist_%d max_hist_%d chrono_%s balance_%s' % \
                (dataset_dir, user_col, '_'.join(skip_cols), df_max_size, min_hist_len, max_hist_len,
                 chrono_split, balance_histories)
    safe_make_dir(cache_dir)

    all_seeds_in_cache = True
    if balance_histories:
        for seed in seeds:
            if not os.path.exists('%s/%d.csv' % (cache_dir, seed)):
                all_seeds_in_cache = False
                break
    else:
        if not os.path.exists('%s/0.csv' % cache_dir):
            all_seeds_in_cache = False

    print('loading data...')
    if not all_seeds_in_cache:
        categ_cols = original_categ_cols.copy()
        try:  # dont one hot encode the user_col
            categ_cols.remove(user_col)
        except ValueError:
            pass

        # load data
        dataset_full = pd.read_csv(dataset_path)
        if df_max_size > 0:
            dataset_full = dataset_full[:df_max_size]
        for col in skip_cols:
            del dataset_full[col]

        # pre-processing for one hot encoding
        col_groups_dict = {}
        categs_unique_values = dataset_full[categ_cols].nunique()
        i = 0
        for col in dataset_full.columns:
            if col in [user_col, target_col]:
                continue
            unique_count = 1
            if col in categ_cols:
                unique_count = categs_unique_values[col]
            col_groups_dict[col] = range(i, i + unique_count)
            i = i + unique_count

        # one hot encoding
        print('one-hot encoding the data... ')
        dataset_full = ce.OneHotEncoder(cols=categ_cols, use_cat_names=True).fit_transform(dataset_full)
        # print('num features = %d' % (dataset_full.shape[1] - 2))  # minus user and target cols

        # splitting histories
        print('balancing and sorting histories...')
        groups_by_user = dataset_full.groupby(user_col, sort=False)
        dataset_full = dataset_full.drop(columns=[user_col])
        all_columns = list(dataset_full.columns)
        if not os.path.exists('%s/all_columns.csv' % cache_dir):
            pd.DataFrame(columns=all_columns).to_csv('%s/all_columns.csv' % cache_dir, index=False)

        del dataset_full

        # get user histories
        for seed in seeds:
            if not os.path.exists('%s/%d.csv' % (cache_dir, seed)):
                hists = {}
                for user_id in groups_by_user.groups.keys():
                    hist = groups_by_user.get_group(user_id).drop(columns=[user_col])
                    if len(hist) < min_hist_len:
                        continue
                    if balance_histories:
                        target_groups = hist.groupby(target_col)
                        if len(target_groups) == 1:  # only one target label present in history: skip
                            continue
                        hist = target_groups.apply(lambda x: x.sample(target_groups.size().min(), random_state=seed))
                        hist.index = hist.index.droplevel(0)
                    hists[user_id] = hist
                # sort hists by len in descending order
                # sorted_hists += [[[k, v] for k, v in reversed(sorted(hists.items(), key=lambda n: len(n[1])))]]
                sorted_hists = [[k, v] for k, v in reversed(sorted(hists.items(), key=lambda n: len(n[1])))]
                seed_df = pd.DataFrame(columns=[user_col] + all_columns, dtype=np.int64)
                for user_id, hist in sorted_hists:
                    hist[user_col] = [user_id] * len(hist)
                    seed_df = seed_df.append(hist, sort=True)
                seed_df.to_csv('%s/%d.csv' % (cache_dir, seed), index=False)
            if not balance_histories:
                break
        del groups_by_user
        del hists
    # end of making seed caches

    # lists indexed by seed containing dicts:
    hist_train_ranges_by_seed = []
    hist_trains_by_seed = []
    hist_valids_by_seed = []
    hist_tests_by_seed = []
    h2_trains_by_seed = []

    user_ids = []
    # min_and_max_feature_values = pd.DataFrame(columns=all_columns, dtype=np.int64)
    min_and_max_feature_values = pd.read_csv('%s/all_columns.csv' % cache_dir, dtype=np.int64)
    all_columns = min_and_max_feature_values.columns
    print('num features = %d' % (len(all_columns) - 1))
    print('splitting histories and composing the general train sets...')
    for seed in seeds:
        # take longest n histories such that train_frac * sum of lens <= h2 train size
        hist_train_ranges = {}
        hist_trains = {}
        hist_valids = {}
        hist_tests = {}
        # one train set per inner seed:
        h2_trains = [pd.DataFrame(columns=all_columns, dtype=np.int64) for i in range(len(inner_seeds))]
        users_checked = 0
        if balance_histories:
            df_seed = pd.read_csv('%s/%d.csv' % (cache_dir, seed))
        else:
            df_seed = pd.read_csv('%s/0.csv' % cache_dir)
        groups_by_user = df_seed.groupby(user_col, sort=False)
        for user_id in groups_by_user.groups.keys():
            hist = groups_by_user.get_group(user_id).drop(columns=[user_col])
            users_checked += 1

            # attempt to add user hist
            hist_len = len(hist)
            if hist_len > max_hist_len:
                # hist is too long: still add user but shorten hist
                if chrono_split:
                    hist = hist[:max_hist_len]
                else:
                    hist = hist.sample(n=max_hist_len, random_state=seed)
                hist_len = max_hist_len

            if min_hist_len <= hist_len * train_frac and len(h2_trains[0]) + hist_len * train_frac <= h2_len:
                if seed == seeds[0]:  # dont add user id more than once
                    user_ids.append(user_id)

                # splitting train, validation and test sets
                hist_train_and_valid_len = int(hist_len * (train_frac + valid_frac)) + 1
                hist_valid_len = int(hist_len * valid_frac) + 1

                # # todo: adjust chrono to inner seed implementation
                # if chrono_split:
                #     hist_train_and_valid = hist[:hist_train_and_valid_len]
                #     hist_valid = hist_train_and_valid[hist_train_and_valid_len - hist_valid_len:]
                # else:
                #     hist_train_and_valid = hist.sample(n=hist_train_and_valid_len, random_state=seed)
                #     hist_valid = hist_train_and_valid.sample(n=hist_valid_len, random_state=seed)
                # hist_train = hist_train_and_valid.drop(hist_valid.index)

                hist_train_and_valid = hist.sample(n=hist_train_and_valid_len, random_state=seed)
                hist_test = hist.drop(hist_train_and_valid.index)
                hist_tests[user_id] = hist_test.reset_index(drop=True)

                hist_trains[user_id] = []
                hist_valids[user_id] = []
                for inner_seed_idx in range(len(inner_seeds)):
                    inner_seed = inner_seeds[inner_seed_idx]
                    hist_valid = hist_train_and_valid.sample(n=hist_valid_len, random_state=inner_seed)
                    hist_train = hist_train_and_valid.drop(hist_valid.index)
                    hist_trains[user_id].append(hist_train.reset_index(drop=True))
                    hist_valids[user_id].append(hist_valid.reset_index(drop=True))
                    h2_train = h2_trains[inner_seed_idx]
                    h2_trains[inner_seed_idx] = h2_train.append(hist_train, sort=True)

                hist_train_ranges[user_id] = [len(h2_train), len(h2_train) + len(hist_train)]

                min_and_max_feature_values = min_and_max_feature_values.append(hist.apply(min_and_max), sort=True)
                if len(h2_train) + min_hist_len * train_frac > h2_len:  # cannot add more users
                    break
        # end of user loop
        hist_trains_by_seed.append(hist_trains)
        hist_valids_by_seed.append(hist_valids)
        hist_tests_by_seed.append(hist_tests)
        h2_trains_by_seed.append([h2_train.reset_index(drop=True) for h2_train in h2_trains])
        hist_train_ranges_by_seed.append(hist_train_ranges)
    # end of seed loop

    print('users = ' + str(len(user_ids)) + ', h2 train len = ' + str(len(h2_train)))

    # fit scaler
    print('fitting scaler...')
    scaler = MinMaxScaler()
    min_and_max_feature_values = min_and_max_feature_values.reset_index(drop=True)
    scaler.fit(min_and_max_feature_values.drop(columns=[target_col]), min_and_max_feature_values[[target_col]])
    labelizer = LabelBinarizer()
    labelizer.fit(min_and_max_feature_values[[target_col]])
    del min_and_max_feature_values

    ccp_alpha_idx = 0
    for ccp_alpha in ccp_alphas:
        ccp_alpha_idx += 1

        # init seed groups
        X_trains_by_seed = []
        Y_trains_by_seed = []
        h1s_by_seed = []
        no_hists_by_seed = []

        print('training h1s and no_hists...')
        for seed_idx in range(len(seeds)):
            seed = seeds[seed_idx]

            X_trains_by_inner_seed = []
            Y_trains_by_inner_seed = []
            h1s_seed = []
            no_hists_seed = []

            for inner_seed_idx in range(len(inner_seeds)):
                inner_seed = inner_seeds[inner_seed_idx]

                # separate train set into X and Y
                h2_train = h2_trains_by_seed[seed_idx][inner_seed_idx]
                X_train = scaler.transform(h2_train.drop(columns=[target_col]))
                Y_train = labelizer.transform(h2_train[[target_col]])

                X_trains_by_inner_seed.append(X_train)
                Y_trains_by_inner_seed.append(Y_train)

                # train h1
                h1 = Models.DecisionTree(X_train[:h1_len], Y_train[:h1_len], 'h1', ccp_alpha, max_depth=max_depth)
                h1s_seed.append(h1)

                if no_hists_by_seed is not None:
                    # train no hists
                    no_hists_by_weight = []
                    for weight in diss_weights:
                        no_hist = Models.ParametrizedTree(X_train, Y_train, ccp_alpha, [1, 1, 0, 0],
                                                          max_depth=max_depth, old_model=h1, diss_weight=weight)
                        no_hists_by_weight.append(no_hist)
                    no_hists_seed.append(no_hists_by_weight)

            # end of inner seed loop
            X_trains_by_seed.append(X_trains_by_inner_seed)
            Y_trains_by_seed.append(Y_trains_by_inner_seed)
            h1s_by_seed.append(h1s_seed)
            no_hists_by_seed.append(no_hists_seed)
        # end of seed loop

        print('\n%d/%d cpp_alpha = %.4f\n' % (ccp_alpha_idx, len(ccp_alphas), ccp_alpha))

        iteration = 0
        iterations = len(user_ids) * len(seeds)
        runtimes = []
        for user_idx, user_id in enumerate(user_ids):
            try:
                last_seed_done = last_seed_done_by_user[user_id]
            except KeyError:
                last_seed_done = -1

            for seed_idx, seed in enumerate(seeds):
                iteration += 1
                if seed <= last_seed_done:
                    continue
                start_time = int(round(time() * 1000))

                # load seed
                X_trains_seed = X_trains_by_seed[seed_idx]
                Y_trains_seed = Y_trains_by_seed[seed_idx]
                h1s_seed = h1s_by_seed[seed_idx]
                no_hists_seed = no_hists_by_seed[seed_idx]

                # with open('%s/valid_log.csv' % result_user_type_dir, 'a', newline='') as log_valid_file:
                #     writer_valid = csv.writer(log_valid_file)
                #     with open('%s/test_log.csv' % result_user_type_dir, 'a', newline='') as log_test_file:
                #         writer_test = csv.writer(log_test_file)

                hist_train_range = hist_train_ranges_by_seed[seed_idx][user_id]
                hist_trains_seed = hist_trains_by_seed[seed_idx][user_id]
                hist_valids_seed = hist_valids_by_seed[seed_idx][user_id]

                # test set
                hist_test = hist_tests_by_seed[seed_idx][user_id]
                hist_test_x = scaler.transform(hist_test.drop(columns=[target_col]))
                hist_test_y = labelizer.transform(hist_test[[target_col]])
                hist_len = len(hist_valids_seed[0]) + len(hist_trains_seed[0]) + len(hist_test)

                # open log files
                files_valid, files_test, writers_valid, writers_test = {}, {}, {}, {}
                for metric in metrics:
                    files_valid[metric] = open('%s/%s/valid_log.csv' % (result_user_type_dir, metric), 'a', newline='')
                    files_test[metric] = open('%s/%s/test_log.csv' % (result_user_type_dir, metric), 'a', newline='')
                    writers_valid[metric] = csv.writer(files_valid[metric])
                    writers_test[metric] = csv.writer(files_test[metric])

                for inner_seed_idx, inner_seed in enumerate(inner_seeds):

                    no_hists_inner_seed = no_hists_seed[inner_seed_idx]
                    X_train, Y_train = X_trains_seed[inner_seed_idx], Y_trains_seed[inner_seed_idx]

                    # train set
                    hist_train = hist_trains_seed[inner_seed_idx]
                    hist_train_x = scaler.transform(hist_train.drop(columns=[target_col]))
                    hist_train_y = labelizer.transform(hist_train[[target_col]])

                    # validation set
                    hist_valid = hist_valids_seed[inner_seed_idx]
                    hist_valid_x = scaler.transform(hist_valid.drop(columns=[target_col]))
                    hist_valid_y = labelizer.transform(hist_valid[[target_col]])

                    # hist
                    hist = Models.History(hist_train_x, hist_train_y)
                    hist.set_range(hist_train_range, len(Y_trains_seed[0]))

                    # h1
                    h1 = h1s_seed[inner_seed_idx]

                    for metric in metrics:
                        h1_acc_valid = h1.test(hist_valid_x, hist_valid_y, metric)['y']
                        h1_acc_test = h1.test(hist_test_x, hist_test_y, metric)['y']

                        for weight_idx, weight in enumerate(diss_weights):
                            row_valid = [user_id, hist_len, seed, inner_seed, h1_acc_valid, weight]
                            row_test = [user_id, hist_len, seed, inner_seed, h1_acc_test, weight]

                            for model_name in model_names:
                                sample_weight = models_to_test[model_name]['sample_weight']
                                if model_name == 'no hist':
                                    h2 = no_hists_inner_seed[weight_idx]
                                else:
                                    h2 = Models.ParametrizedTree(
                                            X_train, Y_train, ccp_alpha, sample_weight, max_depth=max_depth,
                                            old_model=h1, diss_weight=weight, hist=hist)
                                for subset_name in ['valid', 'test']:
                                    x, y = eval('hist_%s_x' % subset_name), eval('hist_%s_y' % subset_name)
                                    result = h2.test(x, y, metric, h1)
                                    row = eval('row_%s' % subset_name)
                                    row.extend([result['x'], result['y']])

                                if plot_confusion:
                                    confusion_dir = '%s/confusion_matrixes/user_%s/seed_%d_%d' % (
                                        result_user_type_dir, user_id, seed, inner_seeds[inner_seed_idx])
                                    if not os.path.exists(confusion_dir):
                                        os.makedirs(confusion_dir)
                                    title = 'user=%s model=%s x=%.2f y=%.2f' % (
                                        user_id, model_name, result['x'], result['y'])
                                    path = '%s/%s_%d' % (confusion_dir, model_name, weight_idx)
                                    plot_confusion_matrix(result['predicted'], hist_test_y, title, path)

                            # end model loop
                            writers_valid[metric].writerow(row_valid)
                            writers_test[metric].writerow(row_test)

                            # end weight loop
                        # end inner seed loop
                for metric in metrics:
                    files_valid[metric].close()
                    files_test[metric].close()

                runtime = (round(time()*1000)-start_time)/1000
                runtimes.append(runtime)
                eta = (iterations - iteration) * np.mean(runtimes)
                print('%d/%d\tuser=%d/%d \tseed=%d/%d \ttime=%.1fs \tETA=%.1fs' %
                      (iteration, iterations, user_idx+1, len(user_ids), seed_idx+1, len(seeds), runtime, eta))
print('\ndone')
