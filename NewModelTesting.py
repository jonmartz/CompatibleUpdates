import csv
import math
import os.path
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
import Models
from sklearn.metrics import auc
import random
import winsound


# todo: L0 model with learned likelihood
# todo: hybrid with multi-label classification
# todo: hist size sensitivity analysis (take one good user and shrink systematically)


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


# Data-set paths

# dataset_name = 'assistment'
# # data settings
# target_col = 'correct'
# original_categ_cols = ['skill', 'tutor_mode', 'answer_type', 'type']
# user_cols = ['user_id']
# # skip_cols = []
# skip_cols = ['skill']
# df_max_size = 100000
# # experiment settings
# train_frac = 0.8
# valid_frac = 0.1
# h1_len = 20
# h2_len = 5000
# seeds = range(10)
# inner_seeds = range(30)
# weights_num = 30
# weights_range = [0, 1]
# sim_ann_var = 0.05
# max_sim_ann_iter = -1
# iters_to_cooling = 100
# # model settings
# max_depth = None
# ccp_alphas = [0.004]
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

dataset_name = "salaries"
# data settings
target_col = 'salary'
original_categ_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                       'native-country']
user_cols = ['relationship']
skip_cols = ['fnlgwt', 'education', 'native-country']
df_max_size = 100000
# experiment settings
train_frac = 0.8
valid_frac = 0.1
h1_len = 20
h2_len = 5000
seeds = range(10)
inner_seeds = range(30, 60)
weights_num = 30
weights_range = [0, 1]
# sim_ann
sim_ann_var = 0.05
max_sim_ann_iter = -1
iters_to_cooling = 100
# model settings
max_depth = None
ccp_alphas = [0.008]
# ccp_alphas = [i / 1000 for i in range(1, 11)]
sample_weights_factor = None
# sample_weights_factor = [0.0, 1.0, 1.0, 1.0]
# best_sample_weight = []
# user settings
min_hist_len = 50
max_hist_len = 1000
current_user_count = 0
users_to_not_test_on = []
only_these_users = []

# dataset_name = "recividism"
# # data settings
# target_col = 'is_recid'
# original_categ_cols = ['sex', 'race', 'age_cat', 'c_charge_degree', 'score_text']
# user_cols = ['race']
# skip_cols = ['c_charge_desc', 'priors_count']
# df_max_size = 100000
# # experiment settings
# train_frac = 0.9
# h1_len = 50
# h2_len = 5000
# seeds = range(10)
# weights_num = 10
# weights_range = [0, 1]
# sim_ann_var = 0.3
# max_sim_ann_iter = -1
# temperature_iters = 100
# # model settings
# max_depth = None
# ccp_alpha = 0.005
# best_sample_weight = [2.849432394, 0.046259433, 2.915855879, 4.184277544]
# # user settings
# min_hist_len = 50
# max_hist_len = 2000
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

# dataset_name = "mooc"
# # data settings
# target_col = 'Opinion(1/0)'
# original_categ_cols = ['course_display_name', 'post_type', 'CourseType']
# user_cols = ['forum_uid']
# skip_cols = ['up_count', 'reads']
# df_max_size = 100000
# # experiment settings
# train_frac = 0.8
# h1_len = 50
# h2_len = 5000
# seeds = range(10)
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

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\mallzee\\mallzee.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\mallzee"
# target_col = 'userResponse'
# categ_cols = ['Currency', 'TypeOfClothing', 'Gender', 'InStock', 'Brand', 'Colour']
# user_group_names = ['userID']
# skip_cols = []
# df_max_size = -1
# layers = []

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\moviesKaggle.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\moviesKaggle"
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

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\titanic\\titanic.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\titanic"
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

# experiment settings
sim_ann = False
chrono_split = False
balance_histories = False

# output settings
make_tradeoff_plots = False
show_tradeoff_plots = False
sound_at_new_best = True

# model settings
models_to_test = {
    # 'no diss': {'sample_weight': [1, 0, 1, 0], 'color': 'grey'},
    'no hist': {'sample_weight': [1, 1, 0, 0], 'color': 'black'},
    # 'sim_ann': {'sample_weight': best_sample_weight, 'color': 'red'},
    'hybrid': {'color': 'green'},
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
model_names_to_test = list(models_to_test.keys())
if not sim_ann:  # only one iteration per model to test
    max_sim_ann_iter = len(model_names_to_test)
if 'hybrid' in model_names_to_test:
    hybrid_range = range(-weights_num, weights_num + 2, 2)
    den = weights_num / 3
    hybrid_weights = list((-i / den for i in hybrid_range))
temperature_delta = 1 / iters_to_cooling
rejects_in_a_row_to_heat = 20

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

dataset_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\%s' % dataset_name
dataset_path = '%s\\%s.csv' % (dataset_dir, dataset_name)

# for ccp_alpha in ccp_alphas:
result_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\current result'
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)
with open('%s\\parameters.csv' % result_dir, 'w', newline='') as file_out:
    writer = csv.writer(file_out)
    writer.writerow(['train_frac', 'ccp_alpha', 'dataset_max_size', 'h1_len', 'h2_len', 'seeds', 'weights_num',
                     'weights_range', 'min_hist_len', 'max_hist_len', 'chrono_split', 'balance_histories', 'skip_cols'])
    writer.writerow([train_frac, str(ccp_alphas), df_max_size, h1_len, h2_len, len(seeds), weights_num,
                     str(weights_range), min_hist_len, max_hist_len, chrono_split, balance_histories, str(skip_cols)])

# run whole experiment for each user column selection
for user_col in user_cols:
    print('USER COLUMN = %s' % user_col)

    # create all folders
    result_user_type_dir = '%s\\%s' % (result_dir, user_col)
    os.makedirs(result_user_type_dir)
    if sim_ann:
        with open(result_user_type_dir + '\\sim_ann.csv', 'w', newline='') as sim_ann_file:
            writer = csv.writer(sim_ann_file)
            writer.writerow(['iteration', 'general_loss', 'general_diss', 'hist_loss', 'hist_diss', 'AUTC', 'accepted'])
    if make_tradeoff_plots:
        os.makedirs('%s\\plots' % result_user_type_dir)

    cache_dir = '%s\\caches\\%s skip_%s max_len_%d min_hist_%d max_hist_%d chrono_%s balance_%s' % \
                (dataset_dir, user_col, '_'.join(skip_cols), df_max_size, min_hist_len, max_hist_len,
                 chrono_split, balance_histories)
    safe_make_dir(cache_dir)

    all_seeds_in_cache = True
    if balance_histories:
        for seed in seeds:
            if not os.path.exists('%s\\%d.csv' % (cache_dir, seed)):
                all_seeds_in_cache = False
                break
    else:
        if not os.path.exists('%s\\0.csv' % cache_dir):
            all_seeds_in_cache = False

    if not all_seeds_in_cache:
        categ_cols = original_categ_cols.copy()
        try:  # dont one hot encode the user_col
            categ_cols.remove(user_col)
        except ValueError:
            pass

        # load data
        print('loading data...')
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
        print('num features = %d' % (dataset_full.shape[1] - 2))  # minus user and target cols

        # splitting histories
        print('balancing and sorting histories...')
        groups_by_user = dataset_full.groupby(user_col, sort=False)
        dataset_full = dataset_full.drop(columns=[user_col])
        all_columns = list(dataset_full.columns)
        if not os.path.exists('%s\\all_columns.csv' % cache_dir):
            pd.DataFrame(columns=all_columns).to_csv('%s\\all_columns.csv' % cache_dir, index=False)

        del dataset_full

        # get user histories
        for seed in seeds:
            if not os.path.exists('%s\\%d.csv' % (cache_dir, seed)):
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
                seed_df.to_csv('%s\\%d.csv' % (cache_dir, seed), index=False)
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

    print('splitting histories and composing the general train sets...')
    user_ids = []
    # min_and_max_feature_values = pd.DataFrame(columns=all_columns, dtype=np.int64)
    min_and_max_feature_values = pd.read_csv('%s\\all_columns.csv' % cache_dir, dtype=np.int64)
    all_columns = min_and_max_feature_values.columns
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
            df_seed = pd.read_csv('%s\\%d.csv' % (cache_dir, seed))
        else:
            df_seed = pd.read_csv('%s\\0.csv' % cache_dir)
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

        no_hists_by_seed = None
        if not sim_ann and 'no hist' in model_names_to_test:
            no_hists_by_seed = []

        print('training h1s and no_hists...')
        for seed_idx in range(len(seeds)):
            seed = seeds[seed_idx]

            X_trains_by_inner_seed = []
            Y_trains_by_inner_seed = []
            h1_by_inner_seed = []

            if no_hists_by_seed is not None:
                no_hists_by_inner_seed = []

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
                h1_by_inner_seed.append(h1)

                if no_hists_by_seed is not None:
                    # train no hists
                    no_hists_by_weight = []
                    for weight in diss_weights:
                        no_hist = Models.ParametrizedTree(X_train, Y_train, ccp_alpha, [1, 1, 0, 0],
                                                          max_depth=max_depth, old_model=h1, diss_weight=weight)
                        no_hists_by_weight.append(no_hist)
                    no_hists_by_inner_seed.append(no_hists_by_weight)

            # end of inner seed loop
            X_trains_by_seed.append(X_trains_by_inner_seed)
            Y_trains_by_seed.append(Y_trains_by_inner_seed)
            h1s_by_seed.append(h1_by_inner_seed)
            if no_hists_by_seed is not None:
                no_hists_by_seed.append(no_hists_by_inner_seed)
        # end of seed loop

        # lists to compute things only once
        h1_acc_valids_by_user_seed = []
        h1_acc_test_by_user_seed = []
        hist_len_by_user_seed = []
        hist_by_user_seed = []
        hist_trains_x_by_user_seed = []
        hist_trains_y_by_user_seed = []
        hist_valids_x_by_user_seed = []
        hist_valids_y_by_user_seed = []
        hist_test_x_by_user_seed = []
        hist_test_y_by_user_seed = []
        h1_avg_acc = None

        # for each model tested
        xs = []
        ys = []
        autcs = []
        xs_no_baseline = []

        # prepare simulated annealing
        if len(only_these_users) > 0:
            user_ids = only_these_users
        sim_ann_iter = 0
        sim_ann_done = False
        autc_prev = None
        sample_weight_prev = None  # [general_loss, general_diss, hist_loss, hist_diss]
        sample_weight_cand = [1, 1, 0, 0]  # start with Ece's method
        best_autc = 0
        initial_temperature = 1
        temperature = initial_temperature
        rejects_in_a_row = 0
        first_sim_ann_iter = True

        if not sim_ann:
            valid_results = pd.DataFrame(columns=['user', 'len', 'seed', 'inner_seed', 'h1_acc', 'weight'])
            test_results = pd.DataFrame(columns=['user', 'len', 'seed', 'inner_seed', 'h1_acc', 'weight'])

        # start simulated annealing
        if sim_ann:
            print('\nstart simulated annealing...')
            print('\tparams=[general_loss, general_diss, hist_loss, hist_diss]')
        else:
            print('\n%d/%d cpp_alpha = %.4f' % (ccp_alpha_idx, len(ccp_alphas), ccp_alpha))

        while not sim_ann_done:  # if not sim_ann (therefore validing models), this loop goes model by model
            start_time = int(round(time.time() * 1000))

            model_valid_results = pd.DataFrame(columns=['len', 'h1_acc', 'weight', 'x', 'y'])
            model_test_results = pd.DataFrame(columns=['len', 'h1_acc', 'weight', 'x', 'y'])

            # getting candidate values
            if sim_ann and not first_sim_ann_iter:
                for i in range(len(sample_weight_cand)):
                    new_val = np.random.normal(sample_weight_prev[i], sim_ann_var, 1)[0]
                    sample_weight_cand[i] = max(0, new_val)
                    # sample_weight_cand[i] = max(0, min(1, new_val))
                    if sample_weights_factor is not None:
                        sample_weight_cand = [sample_weight_cand[j] * sample_weights_factor[j]
                                              for j in range(len(sample_weight_cand))]

            user_count = 0
            user_idx = 0
            iteration = 0
            # iterations = len(user_ids) * len(seeds)
            for user_id in user_ids:
                user_count += 1
                if user_id in users_to_not_test_on or user_count <= current_user_count:
                    iteration += len(seeds)
                    continue

                if first_sim_ann_iter:
                    # lists to save things for seed
                    hist_len_by_seed = []
                    hist_by_seed = []
                    hist_train_x_by_seed = []
                    hist_train_y_by_seed = []
                    hist_valid_x_by_seed = []
                    hist_valid_y_by_seed = []
                    hist_test_x_by_seed = []
                    hist_test_y_by_seed = []
                    h1_acc_valid_by_seed = []
                    h1_acc_test_by_seed = []

                    # save lists for this user
                    hist_len_by_user_seed.append(hist_len_by_seed)
                    hist_by_user_seed.append(hist_by_seed)
                    hist_trains_x_by_user_seed.append(hist_train_x_by_seed)
                    hist_trains_y_by_user_seed.append(hist_train_y_by_seed)
                    hist_valids_x_by_user_seed.append(hist_valid_x_by_seed)
                    hist_valids_y_by_user_seed.append(hist_valid_y_by_seed)
                    hist_test_x_by_user_seed.append(hist_test_x_by_seed)
                    hist_test_y_by_user_seed.append(hist_test_y_by_seed)
                    h1_acc_valids_by_user_seed.append(h1_acc_valid_by_seed)
                    h1_acc_test_by_user_seed.append(h1_acc_test_by_seed)

                for seed_idx in range(len(seeds)):
                    iteration += 1

                    # load seed
                    seed = seeds[seed_idx]
                    X_train_by_inner_seed = X_trains_by_seed[seed_idx]
                    Y_train_by_inner_seed = Y_trains_by_seed[seed_idx]
                    h1_by_inner_seed = h1s_by_seed[seed_idx]
                    if no_hists_by_seed is not None:
                        no_hists_by_inner_seed = no_hists_by_seed[seed_idx]

                    # this if else is to avoid computing things more than once
                    if first_sim_ann_iter:

                        hist_train_range = hist_train_ranges_by_seed[seed_idx][user_id]
                        hist_train_by_inner_seed = hist_trains_by_seed[seed_idx][user_id]
                        hist_valid_by_inner_seed = hist_valids_by_seed[seed_idx][user_id]
                        hist_test = hist_tests_by_seed[seed_idx][user_id]
                        hist_test_x = scaler.transform(hist_test.drop(columns=[target_col]))
                        hist_test_y = labelizer.transform(hist_test[[target_col]])
                        hist_len = len(hist_valid_by_inner_seed[0]) + len(hist_train_by_inner_seed[0]) + len(hist_test)
                        hist_len_by_seed.append(hist_len)

                        hist_train_x_by_inner_seed = []
                        hist_train_y_by_inner_seed = []
                        hist_valid_x_by_inner_seed = []
                        hist_valid_y_by_inner_seed = []
                        hist_by_inner_seed = []
                        h1_acc_valid_by_inner_seed = []
                        h1_acc_test_by_inner_seed = []

                        for inner_seed_idx in range(len(inner_seeds)):
                            # train set by inner seed
                            hist_train = hist_train_by_inner_seed[inner_seed_idx]
                            hist_train_x = scaler.transform(hist_train.drop(columns=[target_col]))
                            hist_train_y = labelizer.transform(hist_train[[target_col]])
                            hist = Models.History(hist_train_x, hist_train_y)
                            hist.set_range(hist_train_range, len(Y_train_by_inner_seed[0]))

                            # valid set by inner seed
                            hist_valid = hist_valid_by_inner_seed[inner_seed_idx]
                            hist_valid_x = scaler.transform(hist_valid.drop(columns=[target_col]))
                            hist_valid_y = labelizer.transform(hist_valid[[target_col]])

                            # h1
                            h1 = h1_by_inner_seed[inner_seed_idx]
                            h1_acc_valid = h1.test(hist_valid_x, hist_valid_y)['auc']
                            h1_acc_test = h1.test(hist_test_x, hist_test_y)['auc']
                            h1_acc_valid_by_inner_seed.append(h1_acc_valid)
                            h1_acc_test_by_inner_seed.append(h1_acc_test)

                            hist_train_x_by_inner_seed.append(hist_train_x)
                            hist_train_y_by_inner_seed.append(hist_train_y)
                            hist_valid_x_by_inner_seed.append(hist_valid_x)
                            hist_valid_y_by_inner_seed.append(hist_valid_y)
                            hist_by_inner_seed.append(hist)

                        hist_by_seed.append(hist_by_inner_seed)
                        hist_train_x_by_seed.append(hist_train_x_by_inner_seed)
                        hist_train_y_by_seed.append(hist_train_y_by_inner_seed)
                        hist_valid_x_by_seed.append(hist_valid_x_by_inner_seed)
                        hist_valid_y_by_seed.append(hist_valid_y_by_inner_seed)
                        h1_acc_valid_by_seed.append(h1_acc_valid_by_inner_seed)
                        h1_acc_test_by_seed.append(h1_acc_test_by_inner_seed)

                        # test set
                        hist_test_x = scaler.transform(hist_test.drop(columns=[target_col]))
                        hist_test_y = labelizer.transform(hist_test[[target_col]])
                        hist_test_x_by_seed.append(hist_test_x)
                        hist_test_y_by_seed.append(hist_test_y)

                    else:
                        hist_len = hist_len_by_user_seed[user_idx][seed_idx]
                        hist_by_inner_seed = hist_by_user_seed[user_idx][seed_idx]
                        hist_valid_x_by_inner_seed = hist_valids_x_by_user_seed[user_idx][seed_idx]
                        hist_valid_y_by_inner_seed = hist_valids_y_by_user_seed[user_idx][seed_idx]
                        h1_acc_valid_by_inner_seed = h1_acc_valids_by_user_seed[user_idx][seed_idx]
                        h1_acc_test_by_inner_seed = h1_acc_test_by_user_seed[user_idx][seed_idx]
                        hist_test_x = hist_test_x_by_user_seed[user_idx][seed_idx]
                        hist_test_y = hist_test_y_by_user_seed[user_idx][seed_idx]

                    # get trade-off
                    weights = diss_weights
                    if not sim_ann:  # only testing models
                        model_name = model_names_to_test[sim_ann_iter]
                        if model_name == 'hybrid':
                            weights = hybrid_weights
                            if no_hists_by_seed is not None:
                                no_hist_valid_by_inner_seed = [no_hists_by_weight[0] for no_hists_by_weight
                                                               in no_hists_by_inner_seed]
                            else:
                                no_hist_valid_by_inner_seed = [
                                    Models.DecisionTree(X_train, Y_train, 'h2', ccp_alpha, max_depth=max_depth,
                                                        old_model=h1, diss_weight=0)
                                    for X_train, Y_train in zip(X_train_by_inner_seed, Y_train_by_inner_seed)
                                ]
                            # todo: copied cached no hists
                            for inner_seed_idx in range(len(inner_seeds)):
                                no_hist_valid = no_hist_valid_by_inner_seed[inner_seed_idx]
                                hist_valid_x = hist_valid_x_by_inner_seed[inner_seed_idx]
                                no_hist_valid.set_hybrid_test(hist, hist_valid_x)

                            no_hist_test_by_inner_seed = [
                                Models.DecisionTree(X_train, Y_train, 'h2', ccp_alpha, max_depth=max_depth,
                                                    old_model=h1, diss_weight=0)
                                for X_train, Y_train, h1
                                in zip(X_train_by_inner_seed, Y_train_by_inner_seed, h1_by_inner_seed)
                            ]
                            for no_hist_test in no_hist_test_by_inner_seed:
                                no_hist_test.set_hybrid_test(hist, hist_test_x)
                        else:
                            sample_weight_cand = models_to_test[model_name]['sample_weight']

                    x_valid_results_by_inner_seed, y_valid_results_by_inner_seed = [], []
                    x_test_results_by_inner_seed, y_test_results_by_inner_seed = [], []
                    for inner_seed in inner_seeds:
                        x_valid_results_by_inner_seed.append([])
                        y_valid_results_by_inner_seed.append([])
                        x_test_results_by_inner_seed.append([])
                        y_test_results_by_inner_seed.append([])
                    for weight_idx in range(len(weights)):
                        weight = weights[weight_idx]
                        if not sim_ann and model_name == 'hybrid':
                            valid_result_by_inner_seed = [no_hist_valid.hybrid_test(hist_valid_y, weight)
                                                          for hist_valid_y, no_hist_valid
                                                          in zip(hist_valid_y_by_inner_seed,
                                                                 no_hist_valid_by_inner_seed)]
                            test_result_by_inner_seed = [no_hist_test.hybrid_test(hist_test_y, weight)
                                                         for no_hist_valid in no_hist_test_by_inner_seed]
                        else:
                            if no_hists_by_seed is not None and model_name == 'no hist':
                                h2_by_inner_seed = [no_hists_by_weight[weight_idx]
                                                    for no_hists_by_weight in no_hists_by_inner_seed]
                            else:
                                h2_by_inner_seed = [
                                    Models.ParametrizedTree(
                                        X_train, Y_train, ccp_alpha, sample_weight_cand,
                                        max_depth=max_depth, old_model=h1, diss_weight=weight, hist=hist)
                                    for X_train, Y_train, h1, hist
                                    in zip(X_train_by_inner_seed, Y_train_by_inner_seed,
                                           h1_by_inner_seed, hist_by_inner_seed)]
                            valid_result_by_inner_seed = [h2.test(hist_valid_x, hist_valid_y, h1)
                                                          for h2, hist_valid_x, hist_valid_y, h1
                                                          in zip(h2_by_inner_seed, hist_valid_x_by_inner_seed,
                                                                 hist_valid_y_by_inner_seed, h1_by_inner_seed)]
                            if not sim_ann:
                                test_result_by_inner_seed = [h2.test(hist_test_x, hist_test_y, h1)
                                                             for h2, h1 in zip(h2_by_inner_seed, h1_by_inner_seed)]

                        for inner_seed_idx in range(len(inner_seeds)):
                            x_valid_results_by_inner_seed[inner_seed_idx].append(
                                valid_result_by_inner_seed[inner_seed_idx]['compatibility'])
                            y_valid_results_by_inner_seed[inner_seed_idx].append(
                                valid_result_by_inner_seed[inner_seed_idx]['auc'])
                            if not sim_ann:
                                x_test_results_by_inner_seed[inner_seed_idx].append(
                                    test_result_by_inner_seed[inner_seed_idx]['compatibility'])
                                y_test_results_by_inner_seed[inner_seed_idx].append(
                                    test_result_by_inner_seed[inner_seed_idx]['auc'])
                    # end of weight loop

                    if not sim_ann and model_name == 'hybrid':
                        weights = list(reversed(weights))

                    seed_valid_results = pd.DataFrame(columns=['len', 'h1_acc', 'weight', 'x', 'y'], dtype=np.int64)
                    for inner_seed_idx in range(len(inner_seeds)):
                        h1_acc_valid = h1_acc_valid_by_inner_seed[inner_seed_idx]
                        seed_valid_results = seed_valid_results.append(
                            pd.DataFrame({'len': hist_len, 'h1_acc': h1_acc_valid, 'weight': weights,
                                          'x': x_valid_results_by_inner_seed[inner_seed_idx],
                                          'y': y_valid_results_by_inner_seed[inner_seed_idx]}))
                    model_valid_results = model_valid_results.append(seed_valid_results)
                    if not sim_ann:
                        seed_test_results = pd.DataFrame(columns=['len', 'h1_acc', 'weight', 'x', 'y'], dtype=np.int64)
                        for inner_seed_idx in range(len(inner_seeds)):
                            h1_acc_test = h1_acc_test_by_inner_seed[inner_seed_idx]
                            seed_test_results = seed_test_results.append(
                                pd.DataFrame(
                                    {'len': hist_len, 'h1_acc': h1_acc_test, 'weight': weights,
                                     'x': x_test_results_by_inner_seed[inner_seed_idx],
                                     'y': y_test_results_by_inner_seed[inner_seed_idx]}))
                        model_test_results = model_test_results.append(seed_test_results)
                        if first_sim_ann_iter:  # prepare general log
                            for inner_seed_idx in range(len(inner_seeds)):
                                inner_seed = inner_seeds[inner_seed_idx]
                                h1_acc_valid = h1_acc_valid_by_inner_seed[inner_seed_idx]
                                h1_acc_test = h1_acc_test_by_inner_seed[inner_seed_idx]
                                seed_valid_results = pd.DataFrame(
                                    {'user': user_id, 'len': hist_len, 'seed': seed, 'inner_seed': inner_seed,
                                     'h1_acc': h1_acc_valid, 'weight': weights})
                                seed_test_results = pd.DataFrame(
                                    {'user': user_id, 'len': hist_len, 'seed': seed, 'inner_seed': inner_seed,
                                     'h1_acc': h1_acc_test, 'weight': weights})
                                valid_results = valid_results.append(seed_valid_results)
                                test_results = test_results.append(seed_test_results)
                # end of seed loop
                user_idx += 1
            # end of user loop
            if not sim_ann:
                valid_results['%s x' % model_name] = model_valid_results['x']
                valid_results['%s y' % model_name] = model_valid_results['y']
                test_results['%s x' % model_name] = model_test_results['x']
                test_results['%s y' % model_name] = model_test_results['y']

            # weighted average over all seeds of all users
            if first_sim_ann_iter:
                h1_avg_acc = np.average(model_valid_results['h1_acc'], weights=model_valid_results['len'])

            sim_ann_iter += 1

            groups = model_valid_results.groupby('weight')
            dfs_by_weight = [groups.get_group(i) for i in groups.groups]
            x = [np.average(i['x'], weights=i['len']) for i in dfs_by_weight]
            y = [np.average(i['y'], weights=i['len']) for i in dfs_by_weight]

            # save values
            if not sim_ann:
                xs.append(x.copy())
                ys.append(y)
                if model_name != 'no diss':
                    xs_no_baseline.append(x.copy())

            # make x monotonic for AUTC
            for i in range(1, len(x)):
                if x[i] < x[i - 1]:
                    x[i] = x[i - 1]
            h1_area = (x[-1] - x[0]) * h1_avg_acc
            autc_cand = auc(x, y) - h1_area

            if first_sim_ann_iter:
                no_hist_autc = autc_cand

            # current improvement
            autc_cand_improv = (autc_cand / no_hist_autc - 1) * 100
            if autc_cand_improv >= 0:
                cand_sign = '+'
            else:
                cand_sign = ''

            if sim_ann:
                # check for new best
                if autc_cand > best_autc:
                    best_autc = autc_cand
                    autc_best_improv = autc_cand_improv
                    if autc_best_improv >= 0:
                        best_sign = '+'
                    else:
                        best_sign = ''
                    if sim_ann and sound_at_new_best:
                        winsound.Beep(800, 500)
                    with open(result_user_type_dir + '\\best_model.txt', 'w', newline='') as sim_ann_file:
                        sim_ann_file.write('%s %s%.4f%%' % (sample_weight_cand, best_sign, autc_best_improv))

                # evaluate candidate
                if first_sim_ann_iter:
                    accept_prob = 1
                else:
                    accept_prob = max(0, min(1, autc_cand / autc_prev))
                    if temperature == 0:
                        if accept_prob < 1:
                            accept_prob = 0
                    elif accept_prob < 1:
                        accept_prob = math.pow(accept_prob, 1 / temperature)
                accepted = False
                if random.uniform(0, 1) < accept_prob:
                    accepted = True
                    autc_prev = autc_cand.copy()
                    sample_weight_prev = sample_weight_cand.copy()
                    rejects_in_a_row = 0
                else:
                    rejects_in_a_row += 1
                    if rejects_in_a_row == rejects_in_a_row_to_heat:
                        temperature = initial_temperature

                with open(result_user_type_dir + '\\sim_ann.csv', 'a', newline='') as sim_ann_file:
                    writer = csv.writer(sim_ann_file)
                    writer.writerow([sim_ann_iter] + sample_weight_cand + [autc_cand, int(accepted)])

            else:
                autcs.append(autc_cand)

            s1, s2, s3, s4 = sample_weight_cand

            if make_tradeoff_plots:
                h1_x = [min(x), max(x)]
                h1_y = [h1_avg_acc, h1_avg_acc]
                plt.plot(h1_x, h1_y, 'k--', marker='.', label='h1')
                plt.plot(x, y, 'b', marker='.', label='h2')
                plt.xlabel('compatibility')
                plt.ylabel('accuracy')
                plt.legend()
                title = 'i=%d [%.4f %.4f %.4f %.4f] autc=%.5f' % (sim_ann_iter, s1, s2, s3, s4, autc_cand)
                if accepted:
                    title += ' accepted'
                else:
                    title += ' rejected'
                plt.title(title)
                plt.savefig('%s\\plots\\%d.png' % (result_user_type_dir, sim_ann_iter))
                if show_tradeoff_plots:
                    plt.show()
                plt.clf()

            runtime = (round(time.time() * 1000) - start_time) / 1000
            if sim_ann:
                print(
                    '%d\tparams=[%.5f %.5f %.5f %.5f] autc %s%.1f%% (best %s%.1f%%) accepted=%s (prob=%.4f) temperature=%.4f time=%.1fs' %
                    (sim_ann_iter, s1, s2, s3, s4, cand_sign, autc_cand_improv, best_sign, autc_best_improv, accepted,
                     accept_prob, temperature, runtime))

                if accepted and temperature > 0:
                    temperature = max(0, temperature - temperature_delta)

            else:  # model validing
                print('%d\tmodel=%s autc=%.5f time=%.1fs' %
                      (sim_ann_iter, model_name, autc_cand, runtime))

            if sim_ann_iter == max_sim_ann_iter:
                sim_ann_done = True

            first_sim_ann_iter = False

        # plot results from model validing
        if not sim_ann:

            valid_results.to_csv('%s\\valid_log.csv' % result_user_type_dir, index=False)
            test_results.to_csv('%s\\test_log.csv' % result_user_type_dir, index=False)

            min_x = min(min(i) for i in xs_no_baseline)  # baseline usually goes backward in compatibility
            h1_x = [f(f(i) for i in xs) for f in [min, max]]  # but still want the plot to look nice
            h1_y = [h1_avg_acc, h1_avg_acc]
            plt.plot(h1_x, h1_y, 'k--', marker='.', label='[g_loss, g_diss, h_loss, h_diss] (h1)')

            i = model_names_to_test.index("no hist")
            x = xs[i]
            y = ys[i]
            no_hist_autc = autcs[i]
            min_x_model = min(x)
            if min_x_model > min_x:  # for models that start at better compatibility
                no_hist_autc += (min_x_model - min_x) * (y[0] - h1_avg_acc)

            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                autc = autcs[i]
                min_x_model = min(x)
                if min_x_model > min_x:  # for models that start at better compatibility
                    autc += (min_x_model - min_x) * (y[0] - h1_avg_acc)
                autc_best_improv = (autc / no_hist_autc - 1) * 100
                if autc_best_improv >= 0:
                    best_sign = '+'
                else:
                    best_sign = ''
                model_name = model_names_to_test[i]
                model = models_to_test[model_name]
                color = model['color']
                if model_name == 'hybrid':
                    label = '                           %s%.1f%% autc (hybrid)' % (best_sign, autc_best_improv)
                elif model_name == 'no diss':
                    s1, s2, s3, s4 = model['sample_weight']
                    label = '[%.1f %.1f %.1f %.1f] (%s)' % (s1, s2, s3, s4, model_name)
                else:
                    s1, s2, s3, s4 = model['sample_weight']
                    label = '[%.1f %.1f %.1f %.1f] %s%.1f%% autc (%s)' % (
                        s1, s2, s3, s4, best_sign, autc_best_improv, model_name)
                plt.plot(x, y, marker='.', label=label, color=color)
            plt.xlabel('compatibility')
            plt.ylabel('accuracy')
            plt.legend(loc='lower left')
            title = 'validing models, dataset=%s, cpp_alpha=%s' % (dataset_name, ccp_alpha)
            plt.title(title)
            plt.savefig('%s\\cpp_alpha_%.5f.png' % (result_user_type_dir, ccp_alpha))
            # if show_tradeoff_plots:
            plt.show()
            plt.clf()
