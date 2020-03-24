import csv
import os.path
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.metrics import confusion_matrix
import seaborn as sn
import Models

# todo: L0 model with nn likelihood
# todo: hybrid with multi-label classification
# todo: hist size sensitivity analysis (take one good user and shrink systematically)


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_monotonic(x, y):
    i = 0
    while i < len(x) - 1:
        if x[i + 1] < x[i]:
            del x[i + 1]
            del y[i + 1]
        else:
            i += 1


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


def get_df_weights(weights, col_groups_dict, seed, model, diss_weight):
    cols = []
    mean_weights = []
    for col_name, group in col_groups_dict.items():
        cols += [col_name]
        max_weight = weights[group].max()
        min_weight = weights[group].min()
        if abs(max_weight) > abs(min_weight):
            mean_weights += [max_weight]
        else:
            mean_weights += [min_weight]
        # mean_weights += [weights[group].mean()]
    return pd.DataFrame({'seed': seed, 'model': model, 'diss_weight': diss_weight, 'col': cols, 'weight': mean_weights})


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


def make_train_plot(user, seed, model, weight, weight_id):
    plot_x = list(range(model.final_epochs))
    # plt.plot(plot_x, model.train_acc, label='train acc')
    # plt.plot(plot_x, model.test_acc, label='test acc')
    plt.plot(plot_x, model.train_loss, label='train loss')
    plt.axvline(model.best_epoch, label='best epoch', color='k', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    # todo: uncomment
    # if only_train_h1:
    #     plt.ylim(bottom, top)
    plt.title('user=%s seed=%d model=%s weight=%.2f\nlayers=%s reg=%.4f'
              % (user, seed, model.model_name, weight, str(layers), regularization))
    path = '%s\\model_training\\user_%s seed_%d model_%s w_%d' % (result_dir, user, seed, model.model_name, weight_id)
    plt.savefig(path)
    if show_train_plots:
        plt.show()
    plt.clf()


# Data-set paths

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\creditRiskAssessment.csv"
# target_col = 'RiskPerformance'

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\recividism\\recividism.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\recividism"
# target_col = 'is_recid'
# original_categ_cols = ['sex', 'race', 'age_cat', 'c_charge_degree', 'score_text']
# user_cols = ['race', 'sex', 'age_cat', 'c_charge_degree', 'score_text']
# skip_cols = ['c_charge_desc', 'priors_count']
# # skip_cols = ['c_charge_desc', 'age_cat', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'score_text', 'decile_score']
# df_max_size = -1
# layers = []

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\fraudDetection\\transactions.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\fraudDetection.csv"
# target_col = 'isFraud'

# dataset_name = 'assistment'
# target_col = 'correct'
# original_categ_cols = ['skill', 'tutor_mode', 'answer_type', 'type']
# user_cols = ['user_id']
# # users_to_not_test_on = problematic_users + [
# #     78917,
# #     78916,
# #     75169,
# #     78905,
# #     78970,
# #     78920,
# #     78912,
# #     78918,
# #     78911,
# #     78921,
# #     78904,
# #     78926,
# #     78919,
# # ]
# users_to_not_test_on = [78534, 72059]
# only_these_users = []
# skip_cols = []
# layers = []
# df_max_size = 100000
# train_frac = 0.8
# h1_len = 200
# h2_len = 5000
# min_h1_epochs = 600
# max_h1_epochs = -1
# min_h2_epochs = 400
# max_h2_epochs = -1
# # acc_tier_height = 0.01
# acc_tier_height = 0.1

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

dataset_name = "salaries"
target_col = 'salary'
original_categ_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                       'native-country']
# user_cols = ['relationship', 'race', 'education', 'occupation', 'marital-status', 'workclass', 'sex', 'native-country']
skip_cols = ['fnlgwt']
user_cols = ['relationship']
users_to_not_test_on = []
only_these_users = []
df_max_size = -1
layers = []
train_frac = 0.8
h1_len = 200
h2_len = 5000
# h1_epochs = 500
# h2_epochs = 200
min_h1_epochs = -1
max_h1_epochs = -1
min_h2_epochs = -1
max_h2_epochs = -1
performance_tier_height = 0.01

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

# dataset_name = 'hospital_mortality'
# target_col = 'HOSPITAL_EXPIRE_FLAG'
# original_categ_cols = ['ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']
# user_cols = ['MARITAL_STATUS']
# users_to_not_test_on = []
# only_these_users = []
# skip_cols = []
# df_max_size = -1
# layers = []
# train_frac = 0.8
# h1_len = 200
# h2_len = 5000
# h1_epochs = 400
# h2_epochs = 200

# selecting experiment parameters

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

# dataset = 'mooc'
# target_col = 'Opinion(1/0)'
# # target_col = 'Question(1/0)'
# # target_col = 'Answer(1/0)'
# original_categ_cols = ['course_display_name', 'post_type', 'CourseType']
# user_cols = ['forum_uid']
# skip_cols = ['up_count', 'reads']
# layers = []
# df_max_size = -1
# train_frac = 0.8
# h1_len = 100
# h2_len = 5000
# h1_epochs = 300
# h2_epochs = 200

# users_to_not_test_on = [0, 18, 5747]

# pre-experiment modes

only_train_h1 = False  # to check how good a standard ML model generalizes

# experiment settings
chrono_split = False
copy_h1_weights = False
balance_histories = True

# experiment scale
seeds = range(3)

# salaries
# diss_weights = [0, 0.2, 1.0]
# diss_weights = [0, 0.1, 0.2, 0.6, 1.0]
diss_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
# diss_weights = [0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1.0]
# assistments
# diss_weights = [0, 0.5, 1.0]
# diss_weights = [0, 0.25, 0.5, 0.75, 1]
# diss_weights = [0, 0.1, 0.25, 0.5, 0.75, 1]
# diss_weights = [0, 0.01, 0.02, 0.06, 0.25, 0.5, 0.75, 1]
# diss_weights = [0, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# automatic diss weights
# diss_count = 5
# diss_weights += [(i + i / (diss_count - 1)) / diss_count for i in range(diss_count)]
# diss_multiply_factor = 1

# model settings
models_to_test = [
    'no hist',
    'L0',
    # 'L1',
    # 'L2',
    'L3',
    'L4',
    'hybrid',
    # 'full_hybrid',
    'baseline',
    # 'adaboost',
    # 'comp_adaboost',
]
batch_size = 128
regularization = 0
normalize_diss_weight = True
only_one_batch = False

# user settings
min_hist_len = 50
max_hist_len = 2000
current_user_count = 0

# plot settings
make_tradeoff_plots = True
make_train_plots = False
show_tradeoff_plots = True
show_train_plots = False
plot_confusion = False

# default settings
range_stds = range(-30, 30, 2)
hybrid_stds = list((-x / 10 for x in range_stds))
colors = {
    'no hist': 'k',
    'L0': 'b',
    'L1': 'yellow',
    'L2': 'purple',
    'L3': 'r',
    'L4': 'purple',
    'hybrid': 'g',
    'full_hybrid': 'c',
    'baseline': 'grey',
    'adaboost': 'blueviolet',
    'comp_adaboost': 'y',
}
hybrid_method = 'nn'

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

# run whole experiment for each user column selection
for user_col in user_cols:
    categ_cols = original_categ_cols.copy()
    try:  # dont one hot encode the user_col
        categ_cols.remove(user_col)
    except ValueError:
        pass

    result_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\current result\\' + user_col
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    else:
        os.makedirs(result_dir)
    os.makedirs(result_dir + '\\by_user_id')
    os.makedirs(result_dir + '\\model_training')
    os.makedirs(result_dir + '\\weights')

    with open(result_dir + '\\log.csv', 'w', newline='') as log_file:
        with open(result_dir + '\\hybrid_log.csv', 'w', newline='') as hybrid_log_file:
            log_writer = csv.writer(log_file)
            hybrid_log_writer = csv.writer(hybrid_log_file)
            log_header = ['train frac', 'user_id', 'instances', 'train seed', 'comp range', 'acc range', 'h1 acc',
                          'diss weight']
            hybrid_log_header = ['train frac', 'user_id', 'instances', 'train seed', 'std offset']
            for name in models_to_test:
                if 'hybrid' not in name:
                    log_header += [name + ' x', name + ' y']
                else:
                    hybrid_log_header += [name + ' x', name + ' y']
            log_writer.writerow(log_header)
            hybrid_log_writer.writerow(hybrid_log_header)

    print('loading data...')
    dataset_full = pd.read_csv(dataset_path)
    if df_max_size >= 0:
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
    print('pre-processing data... ')
    dataset_full = ce.OneHotEncoder(cols=categ_cols, use_cat_names=True).fit_transform(dataset_full)
    print('num features = %d' % (dataset_full.shape[1] - 2))  # minus user and target cols

    # splitting histories
    print('splitting histories into train and test sets...')
    groups_by_user = dataset_full.groupby(user_col)
    dataset_full = dataset_full.drop(columns=[user_col])
    all_columns = list(dataset_full.columns)
    all_dtypes = list(dataset_full.dtypes)
    del dataset_full

    # get user histories
    full_hists = {}
    for user_id in groups_by_user.groups.keys():
        hist = groups_by_user.get_group(user_id).drop(columns=[user_col])
        full_hists[user_id] = hist
    del groups_by_user

    # sort hists by len in descending order
    sorted_full_hists = {k: v for k, v in reversed(sorted(full_hists.items(), key=lambda n: len(n[1])))}
    del full_hists

    # lists indexed by seed containing dicts:
    hist_trains_by_seed = []
    hist_tests_by_seed = []
    h2_train_by_seed = []
    h2_test_by_seed = []

    print('balancing histories and gathering h2 train sets...')
    user_ids = []
    min_and_max_feature_values = pd.DataFrame(columns=all_columns, dtype=np.int64)
    for seed in seeds:
        # take longest n histories such that train_frac * sum of lens <= h2 train size
        hist_trains = {}
        hist_tests = {}
        h2_train = pd.DataFrame(columns=all_columns, dtype=np.int64)
        h2_test = pd.DataFrame(columns=all_columns, dtype=np.int64)
        total_len = 0
        for user_id, hist in sorted_full_hists.items():
            if balance_histories:
                target_groups = hist.groupby(target_col)
                if len(target_groups) == 1:  # only one target label present in history: skip
                    continue
                hist = target_groups.apply(lambda x: x.sample(target_groups.size().min(), random_state=seed))
                hist.index = hist.index.droplevel(0)
            min_and_max_feature_values = min_and_max_feature_values.append(hist.apply(min_and_max))

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

                hist_trains[user_id] = hist_train.reset_index(drop=True)
                hist_tests[user_id] = hist_test.reset_index(drop=True)
                total_len += hist_len
                h2_train = h2_train.append(hist_train)
                h2_test = h2_test.append(hist_test)

                if train_frac * (total_len + min_hist_len) > h2_len:  # cannot add more users
                    break

        hist_trains_by_seed += [hist_trains]
        hist_tests_by_seed += [hist_tests]
        h2_train_by_seed += [h2_train.reset_index(drop=True)]
        h2_test_by_seed += [h2_test.reset_index(drop=True)]
    del sorted_full_hists
    print('users = ' + str(len(user_ids)) + ', h2 train len = ' + str(len(h2_train)))

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
    for model_name in models_to_test:
        if model_name == 'no hist':
            h2s_no_hist_by_seed = []
        if model_name == 'adaboost':
            adaboosts_by_seed = []
        if model_name == 'comp_adaboost':
            comp_adaboosts_by_seed = []

    if only_train_h1:
        print('\nusing cross validation\n')
        bottom, top = 0.4, 1.02

    df_weights = pd.DataFrame(columns=['seed', 'model', 'diss_weight', 'col', 'weight'])

    # train h1 and h2s by seed
    for seed_idx in range(len(seeds)):
        seed = seeds[seed_idx]
        print('\nSETTING TRAIN SEED %d' % seed)

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
        start_time = int(round(time.time() * 1000))
        if only_one_batch:
            batch_size = h1_len
        h1 = Models.NeuralNet(X_train[:h1_len], Y_train[:h1_len], X_test, Y_test, batch_size, layers, 'h1',
                              min_h1_epochs, max_h1_epochs, weights_seed=1, regularization=regularization,
                              performance_tier_size=performance_tier_height)
        runtime = int((round(time.time() * 1000)) - start_time) / 60000
        if only_one_batch:
            batch_size = len(Y_train)

        h1_by_seed += [h1]

        df_weights = df_weights.append(
            get_df_weights(h1.final_weights[0], col_groups_dict, seed, 'h1', 0))

        if make_train_plots:
            make_train_plot('test', seed, h1, 0, 0)

        # train h2s that ignore history
        if not only_train_h1:
            for model_name in models_to_test:
                if model_name == 'no hist':
                    weights = diss_weights
                    h2s_by_seed = h2s_no_hist_by_seed
                elif 'adaboost' in model_name:
                    weights = diss_weights
                    if model_name == 'adaboost':
                        h2s_by_seed = adaboosts_by_seed
                    elif model_name == 'comp_adaboost':
                        h2s_by_seed = comp_adaboosts_by_seed
                else:  # model that needs user history
                    continue
                print('training ' + model_name + ' models...')
                h2s = []
                first_diss_weight = True
                weight_id = 0
                for diss_weight in weights:
                    start_time = int(round(time.time() * 1000))
                    h2 = Models.NeuralNet(X_train, Y_train, X_test, Y_test, batch_size,
                                          layers, model_name, min_h2_epochs, max_h2_epochs, diss_weight=diss_weight,
                                          old_model=h1, copy_h1_weights=copy_h1_weights, weights_seed=2,
                                          regularization=regularization, performance_tier_size=performance_tier_height)
                    runtime = str(int((round(time.time() * 1000)) - start_time) / 1000)
                    print('\tdiss weight ' + str(len(h2s) + 1) + "/" + str(len(weights)) +
                          ' runtime = ' + str(runtime) + 's')
                    h2s += [h2]

                    df_weights = df_weights.append(
                        get_df_weights(h2.final_weights[0], col_groups_dict, seed, model_name, diss_weight))

                    if make_train_plots:
                        make_train_plot('test', seed, h2, diss_weight, weight_id)

                    weight_id += 1

                h2s_by_seed += [h2s]

    df_weights.to_csv('%s\\weights\\no_personalization_weights.csv' % result_dir, index=False)
    if only_train_h1:
        continue

    # test all models on all users for all seeds
    if len(only_these_users) > 0:
        user_ids = only_these_users
    user_count = 0
    iteration = 0
    iterations = len(user_ids) * len(seeds)
    for user_id in user_ids:
        user_count += 1
        if user_id in users_to_not_test_on or user_count <= current_user_count:
            print('\n' + str(iteration) + '/' + str(iterations) + ' ' + user_col + '=' + str(user_id) +
                  ' SKIPPED')
            iteration += len(seeds)
            continue
        for seed_idx in range(len(seeds)):
            iteration += 1

            # load seed
            seed = seeds[seed_idx]
            hist_train = hist_trains_by_seed[seed_idx][user_id]
            hist_test = hist_tests_by_seed[seed_idx][user_id]
            history_len = len(hist_test) + len(hist_train)
            X_train = X_train_by_seed[seed_idx]
            Y_train = Y_train_by_seed[seed_idx]
            X_test = X_test_by_seed[seed_idx]
            Y_test = Y_test_by_seed[seed_idx]
            h1 = h1_by_seed[seed_idx]
            if 'no hist' in models_to_test:
                h2s_no_hist = h2s_no_hist_by_seed[seed_idx]
            if 'adaboost' in models_to_test:
                adaboosts = adaboosts_by_seed[seed_idx]
            if 'comp_adaboost' in models_to_test:
                comp_adaboosts = comp_adaboosts_by_seed[seed_idx]

            if only_one_batch:
                batch_size = len(Y_train)

            # prepare hist
            hist_train_x = scaler.transform(hist_train.drop(columns=[target_col]))
            hist_train_y = labelizer.transform(hist_train[[target_col]])
            hist_test_x = scaler.transform(hist_test.drop(columns=[target_col]))
            hist_test_y = labelizer.transform(hist_test[[target_col]])
            hist = Models.History(hist_train_x, hist_train_y, hist_test_x, hist_test_y)

            print('\n%d/%d %s=%s len=%d seed=%d' % (iteration, iterations, user_col, str(user_id), history_len, seed))

            confusion_dir = result_dir + '\\confusion_matrixes\\' + user_col + '_' + str(user_id) + '\\seed_' + str(
                seed)
            if plot_confusion:
                if not os.path.exists(confusion_dir):
                    os.makedirs(confusion_dir)

            # test h1 on user
            result = h1.test(hist_test_x, hist_test_y)
            h1_acc = result['auc']
            if plot_confusion:
                title = user_col + '=' + str(user_id) + ' h1 y=' + '%.2f' % h1_acc
                path = confusion_dir + '\\h1_seed_' + str(seed) + '.png'
                plot_confusion_matrix(result['predicted'], hist_test_y, title, path)

            # prepare user history for usage
            if 'L0' in models_to_test:
                print('setting likelihoods...')
                hist.set_simple_likelihood(X_train, magnitude_multiplier=1)
            if {'L1', 'L2'}.intersection(set(models_to_test)):
                print('setting kernels...')
                hist.set_kernels(X_train, magnitude_multiplier=10)

            # test all models
            df_weights = pd.DataFrame(columns=['seed', 'model', 'diss_weight', 'col', 'weight'])
            models_x = []
            models_y = []
            for i in range(len(models_to_test)):
                model_name = models_to_test[i]
                print('\tmodel ' + str(i + 1) + "/" + str(len(models_to_test)) + ' = ' + model_name)
                model_x = []
                model_y = []
                models_x += [model_x]
                models_y += [model_y]
                weights = diss_weights

                if 'hybrid' in model_name:
                    weights = hybrid_stds
                    h2 = h2s_no_hist[0]
                    if model_name == 'hybrid':
                        h2.set_hybrid_test(hist, hist_test_x, hybrid_method, layers)
                    elif model_name == 'full_hybrid':
                        h2_train_set = Models.History(X_train, Y_train, X_test, Y_test)
                        h2.set_hybrid_test(h2_train_set, hist_test_x, hybrid_method, layers)
                    df_weights = df_weights.append(
                        get_df_weights(h2.hybrid_feature_weights, col_groups_dict, seed, model_name, 0))

                for j in range(len(weights)):
                    if model_name == 'no hist':
                        result = h2s_no_hist[j].test(hist_test_x, hist_test_y, h1)
                    elif model_name == 'adaboost':
                        result = adaboosts[j].test(hist_test_x, hist_test_y, h1)
                    elif model_name == 'comp_adaboost':
                        result = comp_adaboosts[j].test(hist_test_x, hist_test_y, h1)
                    else:
                        weight = weights[j]
                        if 'hybrid' in model_name:
                            result = h2s_no_hist[0].hybrid_test(hist_test_y, weight)
                        else:
                            start_time = int(round(time.time() * 1000))
                            if weight != 0 or 'no hist' not in models_to_test:
                                h2 = Models.NeuralNet(X_train, Y_train, hist_test_x, hist_test_y, batch_size, layers,
                                                      model_name, min_h2_epochs, max_h2_epochs, diss_weight=weight,
                                                      old_model=h1, history=hist, copy_h1_weights=copy_h1_weights,
                                                      weights_seed=2, regularization=regularization,
                                                      performance_tier_size=performance_tier_height)
                            else:  # no need to train new model if diss_weight = 0 and 'no hist' was already trained
                                h2 = h2s_no_hist[j]
                            runtime = str(int((round(time.time() * 1000)) - start_time) / 1000)
                            print('\t\tweight ' + str(j + 1) + '/' + str(len(weights)) +
                                  ' runtime = ' + str(str(runtime) + 's'))

                            result = h2.test(hist_test_x, hist_test_y, h1)

                            df_weights = df_weights.append(
                                get_df_weights(h2.final_weights[0], col_groups_dict, seed, model_name, weight))

                            if make_train_plots:
                                make_train_plot(str(user_id), seed, h2, weight, j)

                    model_x += [result['compatibility']]
                    model_y += [result['auc']]

                    if plot_confusion:
                        title = user_col + '=' + str(user_id) + ' model=' + model_name \
                                + ' x=' + '%.2f' % (result['compatibility']) + ' y=' + '%.2f' % (result['auc'])
                        path = confusion_dir + '\\' + model_name + '_seed_' + str(seed) + '_' + str(j) + '.png'
                        plot_confusion_matrix(result['predicted'], hist_test_y, title, path)

            df_weights.to_csv('%s\\weights\\%s_weights.csv' % (result_dir, str(user_id)), index=False)

            min_x = min(min(i) for i in models_x)
            min_y = min(min(i) for i in models_y)
            max_x = max(max(i) for i in models_x)
            max_y = max(max(i) for i in models_y)

            com_range = max_x - min_x
            auc_range = max_y - min_y

            if make_tradeoff_plots:

                h1_x = [min_x, max_x]
                h1_y = [h1_acc, h1_acc]
                plt.plot(h1_x, h1_y, 'k--', marker='.', label='h1')

                markersize_delta = 2
                linewidth_delta = 1

                markersize = 8 + markersize_delta * (len(models_to_test) - 1)
                linewidth = 2 + linewidth_delta * (len(models_to_test) - 1)

                for i in range(len(models_to_test)):
                    model_name = models_to_test[i]
                    plt.plot(models_x[i], models_y[i], colors[model_name], marker='.', label=model_name,
                             markersize=markersize, linewidth=linewidth)
                    markersize -= markersize_delta
                    linewidth -= linewidth_delta

                plt.xlabel('compatibility')
                plt.ylabel('accuracy')
                plt.grid()
                plt.legend()
                title = 'user=' + str(user_id) + ' hist_len=' + str(history_len) + ' split=' \
                        + str(train_frac) + ' seed=' + str(seed)
                plt.title(title)
                plt.savefig(
                    result_dir + '\\by_user_id\\' + user_col + '_' + str(user_id) + '_seed_' + str(seed) + '.png')

                if plot_confusion:
                    plt.savefig(confusion_dir + '\\plot.png')
                if show_tradeoff_plots:
                    plt.show()
                plt.clf()

            # write to logs
            has_hybrid = False
            with open(result_dir + '\\log.csv', 'a', newline='') as file_out:
                writer = csv.writer(file_out)
                for i in range(len(diss_weights)):
                    row = [str(train_frac), str(user_id), str(history_len), str(seed),
                           str(com_range), str(auc_range), str(h1_acc), str(diss_weights[i])]
                    for j in range(len(models_to_test)):
                        model_name = models_to_test[j]
                        if 'hybrid' not in model_name:
                            row += [models_x[j][i]]
                            row += [models_y[j][i]]
                        else:
                            has_hybrid = True
                    writer.writerow(row)

            if has_hybrid:
                with open(result_dir + '\\hybrid_log.csv', 'a', newline='') as file_out:
                    writer = csv.writer(file_out)
                    for i in range(len(hybrid_stds)):
                        row = [str(train_frac), str(user_id), str(history_len), str(seed),
                               str(hybrid_stds[i])]
                        for j in range(len(models_to_test)):
                            model_name = models_to_test[j]
                            if 'hybrid' in model_name:
                                row += [models_x[j][i]]
                                row += [models_y[j][i]]
                        writer.writerow(row)
