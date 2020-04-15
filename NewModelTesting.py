import csv
import os.path
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.metrics import confusion_matrix
import seaborn as sn
import Models
from sklearn.metrics import auc
import random


# todo: L0 model with learned likelihood
# todo: hybrid with multi-label classification
# todo: hist size sensitivity analysis (take one good user and shrink systematically)


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


# Data-set paths

# dataset_name = 'assistment'
# # data settings
# target_col = 'correct'
# original_categ_cols = ['skill', 'tutor_mode', 'answer_type', 'type']
# user_cols = ['user_id']
# skip_cols = []
# df_max_size = 100000
# # experiment settings
# train_frac = 0.8
# h1_len = 10
# h2_len = 5000
# seeds = range(5)
# weights_num = 10
# weights_range = [0, 1]
# # model settings
# max_depth = None
# ccp_alphas = []
# ccp_alpha = 0.001
# # user settings
# min_hist_len = 200
# max_hist_len = 2000
# current_user_count = 0
# users_to_not_test_on = []
# only_these_users = []

# dataset_name = "salaries"
# # data settings
# target_col = 'salary'
# original_categ_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                        'native-country']
# user_cols = ['relationship']
# skip_cols = ['fnlgwt']
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
# ccp_alpha = 0.002
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
df_max_size = 100000
# experiment settings
train_frac = 0.8
h1_len = 50
h2_len = 5000
seeds = range(5)
weights_num = 10
weights_range = [0, 1]
sim_ann_var = 0.5
max_sim_ann_iter = -1
# model settings
max_depth = None
ccp_alpha = 0.001
# user settings
min_hist_len = 50
max_hist_len = 2000
current_user_count = 0
users_to_not_test_on = []
only_these_users = []

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

# model settings
# models_to_test = [
#     'no hist',
#     # 'L0',
#     # 'L1',
#     # 'L2',
#     'L3',
#     'L4',
#     'hybrid',
#     # 'full_hybrid',
#     # 'baseline',
#     # 'adaboost',
#     # 'comp_adaboost',
# ]



# experiment settings
sim_ann = False
chrono_split = False
balance_histories = True

# output settings
make_tradeoff_plots = False
show_tradeoff_plots = False
plot_confusion = False
save_logs = False

# model settings
models_to_test = {  # [general_loss, general_diss, hist_loss, hist_diss]
    'baseline': {'sample_weight': [1, 0, 1, 0], 'color': 'grey'},
    'no hist': {'sample_weight': [1, 1, 0, 0], 'color': 'k'},
    'L0': {'sample_weight': [1, 0, 0, 1], 'color': 'r'},
    'L1': {'sample_weight': [1, 1, 0, 1], 'color': 'b'},
    # 'sim_ann': {'sample_weight': [6.196288604, 15.10692767, 0.249410109, 9.585362376], 'color': 'purple'},
    'sim_ann': {'sample_weight': [3.740000035, 0.446685297, 1.897486068, 0.816406222], 'color': 'purple'},
    'hybrid': {'color': 'g'},
    # 'L2': [1, 1, 1, 1],
}

# default settings
diss_weights = np.array([i / weights_num for i in range(weights_num + 1)])
diss_weights = diss_weights * (weights_range[1] - weights_range[0]) + weights_range[0]
print('diss_weights = %s' % diss_weights)
model_names_to_test = list(models_to_test.keys())
if not sim_ann:  # only one iteration per model to test
    max_sim_ann_iter = len(model_names_to_test)
if 'hybrid' in model_names_to_test:
    hybrid_range = range(-weights_num, weights_num + 2, 2)
    den = weights_num / 3
    hybrid_weights = list((-i / den for i in hybrid_range))

# skip cols
user_cols_not_skipped = []
for user_col in user_cols:
    if user_col not in skip_cols:
        user_cols_not_skipped += [user_col]
original_categs_not_skipped = []
for categ in original_categ_cols:
    if categ not in skip_cols:
        original_categs_not_skipped += [categ]
user_cols = user_cols_not_skipped
original_categ_cols = original_categs_not_skipped

dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\%s\\%s.csv' \
               % (dataset_name, dataset_name)

result_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\current result'
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)
with open('%s\\parameters.csv' % result_dir, 'w', newline='') as file_out:
    writer = csv.writer(file_out)
    writer.writerow(['train_frac', 'ccp_alpha', 'dataset_max_size', 'h1_len', 'h2_len', 'seeds', 'weights_num',
                     'weights_range', 'min_hist_len', 'max_hist_len'])
    writer.writerow([train_frac, ccp_alpha, df_max_size, h1_len, h2_len, len(seeds), weights_num,
                     str(weights_range), min_hist_len, max_hist_len])

# run whole experiment for each user column selection
for user_col in user_cols:
    print('USER COLUMN = %s' % user_col)

    categ_cols = original_categ_cols.copy()
    try:  # dont one hot encode the user_col
        categ_cols.remove(user_col)
    except ValueError:
        pass

    # create all folders
    result_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\current result\\' + user_col
    os.makedirs(result_dir)
    with open(result_dir + '\\sim_ann.csv', 'w', newline='') as sim_ann_file:
        writer = csv.writer(sim_ann_file)
        writer.writerow(['iteration', 'general_loss', 'general_diss', 'hist_loss', 'hist_diss', 'AUTC', 'accepted'])
    if make_tradeoff_plots:
        os.makedirs('%s\\plots' % result_dir)
    if save_logs:
        os.makedirs('%s\\logs' % result_dir)

    # if make_tradeoff_plots:
    #     os.makedirs(result_dir + '\\user_plots')
    #
    # with open(result_dir + '\\log.csv', 'w', newline='') as log_file:
    #     with open(result_dir + '\\hybrid_log.csv', 'w', newline='') as hybrid_log_file:
    #         log_writer = csv.writer(log_file)
    #         hybrid_log_writer = csv.writer(hybrid_log_file)
    #         log_header = ['user_id', 'instances', 'train seed', 'comp range', 'acc range', 'h1 acc',
    #                       'diss weight']
    #         hybrid_log_header = ['user_id', 'instances', 'train seed', 'std offset']
    #         for name in models_to_test:
    #             if 'hybrid' not in name:
    #                 log_header += [name + ' x', name + ' y']
    #             else:
    #                 hybrid_log_header += [name + ' x', name + ' y']
    #         log_writer.writerow(log_header)
    #         hybrid_log_writer.writerow(hybrid_log_header)

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
    all_dtypes = list(dataset_full.dtypes)
    del dataset_full

    # get user histories
    sorted_hists = []
    for seed in seeds:
        # print('\tseed %d' % seed)
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
        sorted_hists += [[[k, v] for k, v in reversed(sorted(hists.items(), key=lambda n: len(n[1])))]]
    del groups_by_user
    del hists

    # lists indexed by seed containing dicts:
    hist_train_ranges_by_seed = []
    hist_trains_by_seed = []
    hist_tests_by_seed = []
    h2_train_by_seed = []
    h2_test_by_seed = []

    print('splitting histories and composing the general train sets...')
    user_ids = []
    min_and_max_feature_values = pd.DataFrame(columns=all_columns, dtype=np.int64)
    for seed in seeds:
        # take longest n histories such that train_frac * sum of lens <= h2 train size
        hist_train_ranges = {}
        hist_trains = {}
        hist_tests = {}
        h2_train = pd.DataFrame(columns=all_columns, dtype=np.int64)
        h2_test = pd.DataFrame(columns=all_columns, dtype=np.int64)
        total_len = 0
        users_checked = 0
        for user_id, hist in sorted_hists[seed]:
            users_checked += 1

            # attempt to add user hist
            hist_len = len(hist)
            if hist_len > max_hist_len:
                # hist is too long: still add user but shorten hist
                hist = hist.sample(n=max_hist_len, random_state=seed)
                hist_len = max_hist_len
            if min_hist_len <= hist_len and train_frac * (total_len + hist_len) <= h2_len:
                if chrono_split:
                    hist_train = hist[:int(hist_len * train_frac) + 1]
                else:
                    hist_train = hist.sample(n=int(hist_len * train_frac) + 1, random_state=seed)
                hist_test = hist.drop(hist_train.index)

                # add user hist
                if seed == seeds[0]:  # dont add user id more than once
                    user_ids += [user_id]

                hist_train_ranges[user_id] = [len(h2_train), len(h2_train) + len(hist_train)]  # hist range in h2 train
                hist_trains[user_id] = hist_train.reset_index(drop=True)
                hist_tests[user_id] = hist_test.reset_index(drop=True)
                total_len += hist_len
                h2_train = h2_train.append(hist_train)
                h2_test = h2_test.append(hist_test)
                min_and_max_feature_values = min_and_max_feature_values.append(hist.apply(min_and_max))

                if train_frac * (total_len + min_hist_len) > h2_len:  # cannot add more users
                    break
        # print('\tseed %d users checked = %d' % (seed, users_checked))

        hist_train_ranges_by_seed += [hist_train_ranges]
        hist_trains_by_seed += [hist_trains]
        hist_tests_by_seed += [hist_tests]
        h2_train_by_seed += [h2_train.reset_index(drop=True)]
        h2_test_by_seed += [h2_test.reset_index(drop=True)]

    del sorted_hists
    print('users = ' + str(len(user_ids)) + ', h2 train len = ' + str(len(h2_train)))

    # fit scaler
    print('fitting scaler...')
    scaler = MinMaxScaler()
    min_and_max_feature_values = min_and_max_feature_values.reset_index(drop=True)
    scaler.fit(min_and_max_feature_values.drop(columns=[target_col]), min_and_max_feature_values[[target_col]])
    labelizer = LabelBinarizer()
    labelizer.fit(min_and_max_feature_values[[target_col]])
    del min_and_max_feature_values

    # init seed groups
    h1_by_seed = []
    X_train_by_seed = []
    Y_train_by_seed = []
    X_test_by_seed = []
    Y_test_by_seed = []

    print('splitting train and test sets into x and y...')
    for seed_idx in range(len(seeds)):
        seed = seeds[seed_idx]

        # separate train set into X and Y
        h2_train = h2_train_by_seed[seed_idx]
        h2_test = h2_test_by_seed[seed_idx]
        X_train = scaler.transform(h2_train.drop(columns=[target_col]))
        Y_train = labelizer.transform(h2_train[[target_col]])
        X_test = scaler.transform(h2_test.drop(columns=[target_col]))
        Y_test = labelizer.transform(h2_test[[target_col]])

        X_train_by_seed += [X_train]
        Y_train_by_seed += [Y_train]
        X_test_by_seed += [X_test]
        Y_test_by_seed += [Y_test]

        # train h1
        h1 = Models.DecisionTree(X_train[:h1_len], Y_train[:h1_len], 'h1', ccp_alpha, max_depth=max_depth)
        h1_by_seed += [h1]

    # lists to compute things only once
    h1_acc_by_user_seed = []
    hist_len_by_user_seed = []
    hist_by_user_seed = []
    hist_test_x_by_user_seed = []
    hist_test_y_by_user_seed = []
    h1_avg_acc = None

    # for each model tested
    xs = []
    ys = []
    autcs = []

    # prepare simulated annealing
    if len(only_these_users) > 0:
        user_ids = only_these_users
    sim_ann_iter = 0
    sim_ann_done = False
    autc_prev = None
    sample_weight_prev = None  # [general_loss, general_diss, hist_loss, hist_diss]
    sample_weight_cand = [1, 1, 0, 0]  # start with Ece's method

    # start simulated annealing
    if sim_ann:
        print('\nstart simulated annealing...')
        print('\tparams=[general_loss, general_diss, hist_loss, hist_diss]')
    else:
        print('\ntesting models...')
    while not sim_ann_done:
        start_time = int(round(time.time() * 1000))
        df_results = pd.DataFrame(columns=['user', 'len', 'seed', 'h1_acc', 'weight', 'x', 'y'])

        # getting candidate values
        if sim_ann and autc_prev is not None:
            for i in range(len(sample_weight_cand)):
                sample_weight_cand[i] = max(0, np.random.normal(sample_weight_prev[i], sim_ann_var, 1)[0])

        user_count = 0
        user_idx = 0
        iteration = 0
        # iterations = len(user_ids) * len(seeds)
        for user_id in user_ids:
            user_count += 1
            if user_id in users_to_not_test_on or user_count <= current_user_count:
                iteration += len(seeds)
                continue

            if sim_ann_iter == 0:

                # lists to save things for seed
                hist_len_by_seed = []
                hist_by_seed = []
                hist_test_x_by_seed = []
                hist_test_y_by_seed = []
                h1_acc_by_seed = []

                # save lists for this user
                hist_len_by_user_seed.append(hist_len_by_seed)
                hist_by_user_seed.append(hist_by_seed)
                hist_test_x_by_user_seed.append(hist_test_x_by_seed)
                hist_test_y_by_user_seed.append(hist_test_y_by_seed)
                h1_acc_by_user_seed.append(h1_acc_by_seed)

            for seed_idx in range(len(seeds)):
                iteration += 1

                # load seed
                seed = seeds[seed_idx]
                X_train = X_train_by_seed[seed_idx]
                Y_train = Y_train_by_seed[seed_idx]
                h1 = h1_by_seed[seed_idx]

                # this if else is to avoid computing things more than once
                if sim_ann_iter == 0:
                    hist_train_range = hist_train_ranges_by_seed[seed_idx][user_id]
                    hist_train = hist_trains_by_seed[seed_idx][user_id]
                    hist_test = hist_tests_by_seed[seed_idx][user_id]
                    # X_test = X_test_by_seed[seed_idx]
                    # Y_test = Y_test_by_seed[seed_idx]
                    hist_train_x = scaler.transform(hist_train.drop(columns=[target_col]))
                    hist_train_y = labelizer.transform(hist_train[[target_col]])
                    hist_len = len(hist_test) + len(hist_train)
                    hist = Models.History(hist_train_x, hist_train_y)
                    hist.set_range(hist_train_range, len(Y_train))
                    hist_test_x = scaler.transform(hist_test.drop(columns=[target_col]))
                    hist_test_y = labelizer.transform(hist_test[[target_col]])
                    h1_acc = h1.test(hist_test_x, hist_test_y)['auc']

                    # save things in seed's list
                    hist_len_by_seed.append(hist_len)
                    hist_by_seed.append(hist)
                    hist_test_x_by_seed.append(hist_test_x)
                    hist_test_y_by_seed.append(hist_test_y)
                    h1_acc_by_seed.append(h1_acc)
                else:
                    hist_len = hist_len_by_user_seed[user_idx][seed_idx]
                    hist = hist_by_user_seed[user_idx][seed_idx]
                    hist_test_x = hist_test_x_by_user_seed[user_idx][seed_idx]
                    hist_test_y = hist_test_y_by_user_seed[user_idx][seed_idx]
                    h1_acc = h1_acc_by_user_seed[user_idx][seed_idx]

                # if plot_confusion:
                #     title = user_col + '=' + str(user_id) + ' h1 y=' + '%.2f' % h1_acc
                #     path = confusion_dir + '\\h1_seed_' + str(seed) + '.png'
                #     plot_confusion_matrix(result['predicted'], hist_test_y, title, path)

                # prepare user history for usage
                # if 'L0' in models_to_test:
                #     print('setting likelihoods...')
                #     hist.set_simple_likelihood(X_train, magnitude_multiplier=1)
                # if {'L1', 'L2'}.intersection(set(models_to_test)):
                #     print('setting kernels...')
                #     hist.set_kernels(X_train, magnitude_multiplier=10)

                # # test all models
                # models_x = []
                # models_y = []

                # for i in range(len(models_to_test)):
                #     model_name = models_to_test[i]
                #     model_x, model_y = [], []
                #     models_x.append(model_x)
                #     models_y.append(model_y)
                #     weights = diss_weights
                #
                #     if 'hybrid' in model_name:
                #         weights = hybrid_stds
                #         h2 = h2s_no_hist[0]
                #         if model_name == 'hybrid':
                #             history_for_hybrid = hist
                #         elif model_name == 'full_hybrid':
                #             history_for_hybrid = Models.History(X_train, Y_train, X_test, Y_test)
                #         h2.set_hybrid_test(history_for_hybrid, hist_test_x)
                #
                #     start_time = int(round(time.time() * 1000))
                #     for j in range(len(weights)):
                #         if model_name == 'no hist':
                #             result = h2s_no_hist[j].test(hist_test_x, hist_test_y, h1)
                #         elif model_name == 'adaboost':
                #             result = adaboosts[j].test(hist_test_x, hist_test_y, h1)
                #         elif model_name == 'comp_adaboost':
                #             result = comp_adaboosts[j].test(hist_test_x, hist_test_y, h1)
                #         else:
                #             weight = weights[j]
                #             if 'hybrid' in model_name:
                #                 result = h2s_no_hist[0].hybrid_test(hist_test_y, weight)
                #             else:
                #                 # start_time = int(round(time.time() * 1000))
                #                 if ('adaboost' not in model_name and weight > 0) or 'no hist' not in models_to_test:
                #                     h2 = Models.DecisionTree(X_train, Y_train, model_name, ccp_alpha, hist_test_x,
                #                                              hist_test_y, old_model=h1, diss_weight=weight, hist=hist)
                #                 else:  # no need to train new model if diss_weight = 0 and 'no hist' was already trained
                #                     h2 = h2s_no_hist[j]
                #                 result = h2.test(hist_test_x, hist_test_y, h1)
                #         model_x += [result['compatibility']]
                #         model_y += [result['auc']]
                #
                #         if plot_confusion:
                #             title = user_col + '=' + str(user_id) + ' model=' + model_name \
                #                     + ' x=' + '%.2f' % (result['compatibility']) + ' y=' + '%.2f' % (result['auc'])
                #             path = confusion_dir + '\\' + model_name + '_seed_' + str(seed) + '_' + str(j) + '.png'
                #             plot_confusion_matrix(result['predicted'], hist_test_y, title, path)
                #
                #     runtime = str(int((round(time.time() * 1000)) - start_time) / 1000)
                #     print('\t%d/%d %s %ss' % (i + 1, len(models_to_test), model_name, str(runtime)))

                # get trade-off plot
                weights = diss_weights
                if not sim_ann:  # only testing models
                    model_name = model_names_to_test[sim_ann_iter]
                    if model_name == 'hybrid':
                        weights = hybrid_weights
                        h2_no_hist = Models.DecisionTree(X_train, Y_train, 'h2', ccp_alpha, max_depth=max_depth,
                                                         old_model=h1, diss_weight=0)
                        h2_no_hist.set_hybrid_test(hist, hist_test_x)
                    else:
                        sample_weight_cand = models_to_test[model_name]['sample_weight']

                model_x, model_y = [], []
                for j in range(len(weights)):
                    weight = weights[j]
                    if not sim_ann and model_name == 'hybrid':
                        result = h2_no_hist.hybrid_test(hist_test_y, weight)
                    else:
                        h2 = Models.ParametrizedTree(X_train, Y_train, ccp_alpha, sample_weight_cand,
                                                     max_depth=max_depth, old_model=h1, diss_weight=weight, hist=hist)
                        result = h2.test(hist_test_x, hist_test_y, h1)
                    model_x += [result['compatibility']]
                    model_y += [result['auc']]
                if not sim_ann and model_name == 'hybrid':
                    weights = reversed(weights)
                df_results = df_results.append(
                    pd.DataFrame({'user': user_id, 'len': hist_len, 'seed': seed, 'h1_acc': h1_acc,
                                  'weight': weights, 'x': model_x, 'y': model_y}))
            user_idx += 1

        # finishing this sim_ann_iter
        if save_logs:
            df_results.to_csv('%s\\logs\\%d.csv' % (result_dir, sim_ann_iter), index=False)

        # weighted average over all seeds of all users
        if sim_ann_iter == 0:
            h1_avg_acc = np.average(df_results['h1_acc'], weights=df_results['len'])

        sim_ann_iter += 1

        groups = df_results.groupby('weight')
        dfs_by_weight = [groups.get_group(i) for i in groups.groups]
        x = [np.average(i['x'], weights=i['len']) for i in dfs_by_weight]
        y = [np.average(i['y'], weights=i['len']) for i in dfs_by_weight]

        # save values
        if not sim_ann:
            xs.append(x.copy())
            ys.append(y)

        # make x monotonic for AUTC
        for i in range(1, len(x)):
            if x[i] < x[i - 1]:
                x[i] = x[i - 1]
        h1_area = (x[-1] - x[0]) * h1_avg_acc
        autc_cand = auc(x, y) - h1_area

        if not sim_ann:
            autcs.append(autc_cand)

        # evaluate candidate
        accepted = False
        if autc_prev is None or random.uniform(0, 1) < min(1, autc_cand / autc_prev):
            accepted = True
            autc_prev = autc_cand.copy()
            sample_weight_prev = sample_weight_cand

        with open(result_dir + '\\sim_ann.csv', 'a', newline='') as sim_ann_file:
            writer = csv.writer(sim_ann_file)
            writer.writerow([sim_ann_iter] + sample_weight_cand + [autc_cand, int(accepted)])

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
            plt.savefig('%s\\plots\\%d.png' % (result_dir, sim_ann_iter))
            if show_tradeoff_plots:
                plt.show()
            plt.clf()

        runtime = (round(time.time() * 1000) - start_time) / 1000
        print('%d\tparams=[%.5f %.5f %.5f %.5f] autc=%.5f accepted=%s time=%.1fs' %
              (sim_ann_iter, s1, s2, s3, s4, autc_cand, accepted, runtime))

        if sim_ann_iter == max_sim_ann_iter:
            sim_ann_done = True

    # test models
    if not sim_ann:
        h1_x = [min(min(i) for i in xs), max(max(i) for i in xs)]
        h1_y = [h1_avg_acc, h1_avg_acc]
        plt.plot(h1_x, h1_y, 'k--', marker='.', label='[t_loss, t_diss, h_loss, h_diss] (h1)')

        no_hist_idx = model_names_to_test.index("no hist")
        no_hist_autc = autcs[no_hist_idx]

        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            autc = autcs[i]
            autc_improv = (autc / no_hist_autc - 1) * 100
            if autc_improv >= 0:
                sign = '+'
            else:
                sign = ''
            model_name = model_names_to_test[i]
            model = models_to_test[model_name]
            color = model['color']
            if model_name == 'hybrid':
                label = '                           %s%.1f%% autc (hybrid)' % (sign, autc_improv)
            elif model_name == 'baseline':
                s1, s2, s3, s4 = model['sample_weight']
                label = '[%.1f %.1f %.1f %.1f] (%s)' % ( s1, s2, s3, s4, model_name)
            else:
                s1, s2, s3, s4 = model['sample_weight']
                label = '[%.1f %.1f %.1f %.1f] %s%.1f%% autc (%s)' % (s1, s2, s3, s4, sign, autc_improv, model_name)
            plt.plot(x, y, marker='.', label=label, color=color)
        plt.xlabel('compatibility')
        plt.ylabel('accuracy')
        plt.legend()
        title = 'testing models, dataset=%s' % dataset_name
        plt.title(title)
        plt.savefig('%s\\testing_models.png' % result_dir)
        # if show_tradeoff_plots:
        plt.show()
        plt.clf()
