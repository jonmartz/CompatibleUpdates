import csv
import os
from time import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from ModelTesting import get_time_string
import itertools
import random


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


def get_feat_importance(x, y):
    tree = DecisionTreeRegressor(random_state=1)
    tree.fit(x, y)
    y_pred = np.round(tree.predict(x))
    accuracy = np.mean(np.equal(y_pred, y).astype(int))
    feat_importance = tree.feature_importances_
    return feat_importance, accuracy


def get_cache_and_params(log_dir, user_col):
    params_path = '/'.join(log_dir.split('/')[:-2]) + '/parameters.csv'
    params = pd.read_csv(params_path)
    row = params.iloc[0]
    cache = '%s skip_%s max_len_%d min_hist_%d max_hist_%d balance_%s' % (
        # user_col, '_'.join(eval(row['skip_cols'])), row['dataset_max_size'], row['min_hist_len'], row['max_hist_len'],
        user_col, len(eval(row['skip_cols'])), row['dataset_max_size'], row['min_hist_len'], row['max_hist_len'],
        row['balance_histories']
    )
    return cache, params


def make_data_analysis(log_dir, dataset, user_col, target_col):
    if os.path.exists('%s/wasserstein_distances.csv' % log_dir):
        return

    cache, params = get_cache_and_params(log_dir, user_col)
    dataset_dir = 'C:/Users/Jonma/Documents/BGU/Thesis/DataSets/%s/caches/%s' % (dataset, cache)
    dataset_path = '%s/0.csv' % dataset_dir
    df = pd.read_csv(dataset_path)

    cols = list(df.drop(columns=[user_col, target_col]).columns) + [target_col]  # make sure target_col is the last col
    df = df[[user_col] + cols]  # make sure user_col is the first col

    scaler = MinMaxScaler()
    df_x = df.drop(columns=[target_col, user_col])
    cols_no_target = df_x.columns
    y = df[target_col]
    x_norm = scaler.fit_transform(df_x, y)
    df_norm = pd.DataFrame(x_norm, columns=df_x.columns)
    df_norm[target_col] = y
    df_norm[user_col] = df[user_col]

    gen_feat_importance, gen_acc = get_feat_importance(df_norm.drop(columns=[target_col, user_col]), df_norm[target_col])
    print('gen acc = %.5f' % gen_acc)
    users = list(pd.unique(df_norm[user_col]))
    user_groups = df_norm.groupby(user_col)
    wasserstein_distances = [[] for i in cols]
    # cosine_distances = [[] for i in cols]
    feat_importances = [[gen_feat_importance[i]] for i in range(len(cols_no_target))]
    for user in users:
        df_user = user_groups.get_group(user)
        for i in range(len(cols)):
            col = cols[i]
            wasserstein_distances[i].append(wasserstein_distance(df_norm[col], df_user[col]))
            # cosine_distances[i].append(1 - cosine(df_norm[col], df_user[col]))
        user_feat_importance, user_acc = get_feat_importance(df_user.drop(columns=[target_col, user_col]),
                                                             df_user[target_col])
        print('\tuser %s acc = %.5f' % (user, user_acc))
        for i in range(len(cols_no_target)):
            feat_importances[i].append(user_feat_importance[i])

    # write distances
    df_dict = {'user': users}
    for i in range(len(cols)):
        df_dict[cols[i]] = wasserstein_distances[i]
    pd.DataFrame(df_dict).to_csv('%s/wasserstein_distances.csv' % log_dir, index=False)

    # write feature importances
    df_dict = {'user': ['general'] + users}
    for i in range(len(cols_no_target)):
        df_dict[cols_no_target[i]] = feat_importances[i]
    df_dict[target_col] = [1] * (len(users) + 1)
    pd.DataFrame(df_dict).to_csv('%s/feature_importances.csv' % log_dir, index=False)


def make_data_analysis_per_inner_seed(log_dir, dataset, user_col, target_col, min_hist_len_to_test,
                                      file_name='user_features', num_top_features=5, chrono_split=True):
    if os.path.exists('%s/%s_by_seed.csv' % (log_dir, file_name)):
        return
    mode = 'w'
    done_rows = {}
    if os.path.exists('%s/%s.csv' % (log_dir, file_name)):
        mode = 'a'
        existing_df = pd.read_csv('%s/%s.csv' % (log_dir, file_name))
        for i, row in existing_df.iterrows():
            user, seed, inner_seed = row['user'], row['seed'], row['inner_seed']
            if seed not in done_rows:
                seed_dict = {}
                done_rows[seed] = seed_dict
            else:
                seed_dict = done_rows[seed]
            if inner_seed not in seed_dict:
                inner_seed_set = set()
                seed_dict[inner_seed] = inner_seed_set
            else:
                inner_seed_set = seed_dict[inner_seed]
            inner_seed_set.add(user)

    # get cached dataset
    cache, params = get_cache_and_params(log_dir, user_col)
    needed_params = params[['max_hist_len', 'seeds', 'inner_seeds', 'train_frac', 'valid_frac']].iloc[0]
    max_hist_len, num_seeds, num_inner_seeds, train_frac, valid_frac = [float(i) for i in needed_params]
    max_hist_len = int(max_hist_len)
    seeds, inner_seeds = list(range(int(num_seeds))), list(range(int(num_inner_seeds)))
    cache_dir = 'C:/Users/Jonma/Documents/BGU/Thesis/DataSets/%s/caches/%s' % (dataset, cache)

    print("loading users...")
    all_cols = pd.read_csv('%s/all_columns.csv' % cache_dir).columns
    features = list(all_cols.drop(target_col))
    dataset = pd.read_csv('%s/0.csv' % cache_dir)
    users_col = dataset[user_col]

    # normalize dataset
    dataset = MinMaxScaler().fit_transform(dataset.drop(columns=user_col))
    dataset = pd.DataFrame(dataset, columns=all_cols)

    # get top n features
    baseline = DecisionTreeRegressor(random_state=1, ccp_alpha=0)  # todo: maybe set to ccp_alpha
    baseline.fit(dataset.drop(columns=target_col), dataset[target_col])
    feature_importances = baseline.feature_importances_
    sorted_features = [i for _, i in reversed(sorted(zip(feature_importances, features)))]
    top_features = sorted_features[:num_top_features]

    dataset.insert(0, user_col, users_col)
    del users_col

    groups_by_user = dataset.groupby(user_col, sort=False)
    hists_by_user = {}
    hist_train_lens = {}
    num_users_to_test = 0
    for user_id, hist in groups_by_user:
        hist = hist.drop(columns=[user_col])
        if len(hist) > max_hist_len:
            hist = hist[:max_hist_len]
        if len(hist) >= min_hist_len_to_test:
            num_users_to_test += 1
        hist_train_lens[user_id] = int(len(hist) * train_frac)
        hists_by_user[user_id] = hist
    del groups_by_user

    print('start analysis')
    iterations = len(seeds) * len(inner_seeds) * num_users_to_test
    iteration = 0
    avg_runtime = 0
    num_runtimes = 0
    with open('%s/%s.csv' % (log_dir, file_name), mode=mode, newline='') as file:
        writer = csv.writer(file)

        if mode == 'w':
            row = ['user', 'len', 'seed', 'inner_seed']
            for i, j in itertools.combinations(['all_train', 'hist_train', 'hist_valid', 'hist_test'], 2):
                row.append('%s to %s' % (i, j))
            writer.writerow(row + top_features)

        for seed_idx, seed in enumerate(seeds):

            if seed in done_rows and len(done_rows[seed]) == len(inner_seeds):
                seed_dict = done_rows[seed]
                if all([len(seed_dict[i]) == len(hists_by_user) for i in inner_seeds]):
                    iteration += len(inner_seeds) * len(hists_by_user)
                    continue

            # split the test sets
            hists_seed_by_user = {}
            for user_idx, item in enumerate(hists_by_user.items()):
                user_id, hist = item
                if chrono_split:
                    valid_len = int(len(hist) * valid_frac)
                    test_len = int(len(hist) * (1 - (train_frac + valid_frac)))
                    min_idx = 3 * valid_len  # |train set| >= 2|valid set|
                    delta = len(hist) - test_len - min_idx  # space between min_idx and test_start_idx
                    delta_frac = list(np.linspace(1, 0, len(seeds)))
                    random.seed(user_idx)
                    random.shuffle(delta_frac)
                    test_start_idx = min_idx + int(delta * delta_frac[seed])
                    hist_train_and_valid = hist.iloc[0: test_start_idx]
                    hist_test = hist.iloc[test_start_idx: test_start_idx + test_len + 1]
                else:
                    hist_train_and_valid = hist.sample(n=int(len(hist) * (train_frac + valid_frac)), random_state=seed)
                    hist_test = hist.drop(hist_train_and_valid.index).reset_index(drop=True)
                hists_seed_by_user[user_id] = [hist_train_and_valid, hist_test.drop(columns=target_col)]

            # run analysis for each inner seed
            for inner_seed_idx, inner_seed in enumerate(inner_seeds):

                if seed in done_rows and inner_seed in done_rows[seed]:
                    if len(done_rows[seed][inner_seed]) == len(hists_by_user):
                        iteration += len(hists_by_user)
                        continue

                hists_inner_seed_by_user = {}
                h2_train = pd.DataFrame(columns=all_cols, dtype=np.float32)

                for user_id, item in hists_seed_by_user.items():
                    hist_train_and_valid, hist_test = item
                    hist_train_len = hist_train_lens[user_id]
                    if chrono_split:
                        hist_len = len(hists_by_user[user_id])  # todo: maybe this is wrong
                        valid_len = int(hist_len * valid_frac)
                        delta = len(hist_train_and_valid) - 2 * valid_len  # space between min_idx and valid_start
                        delta_frac = list(np.linspace(1, 0, len(inner_seeds)))
                        random.seed(user_idx)
                        random.shuffle(delta_frac)
                        valid_start_idx = valid_len + int(delta * delta_frac[inner_seed])
                        hist_train = hist_train_and_valid.iloc[0: valid_start_idx]
                        hist_valid = hist_train_and_valid.iloc[valid_start_idx:]
                    else:
                        hist_train = hist_train_and_valid.sample(n=hist_train_len, random_state=inner_seed)
                        hist_valid = hist_train_and_valid.drop(hist_train.index)
                    h2_train = h2_train.append(hist_train, ignore_index=True, sort=False)
                    hists_inner_seed_by_user[user_id] = [hist_train.drop(columns=target_col),
                                                         hist_valid.drop(columns=target_col), hist_test]

                h2_train_x = h2_train.drop(columns=[target_col])
                baseline = DecisionTreeRegressor(random_state=1, ccp_alpha=0)  # todo: maybe set to ccp_alpha
                baseline.fit(h2_train_x, h2_train[target_col])
                feature_importances = baseline.feature_importances_

                user_count = 0
                for user_id, item in hists_inner_seed_by_user.items():

                    hist_train, hist_valid, hist_test = item
                    hist_len = len(hist_train) + len(hist_valid) + len(hist_test)
                    # if hist_len < min_hist_len_to_test:
                    #     break

                    iteration += 1
                    user_count += 1

                    if seed in done_rows and inner_seed in done_rows[seed] and user_id in done_rows[seed][inner_seed]:
                        continue

                    start_time = int(round(time() * 1000))

                    row = [user_id, hist_len, seed, inner_seed]
                    # add wasserstein distances
                    for i, j in itertools.combinations([h2_train_x, hist_train, hist_valid, hist_test], 2):
                        dist = [wasserstein_distance(i[col], j[col]) for col in features]
                        row.append(np.average(dist, weights=feature_importances))
                    # add selected domain specific features
                    selected_feature_values = list(hist_train[top_features].mean())
                    writer.writerow(row + selected_feature_values)

                    runtime = (round(time() * 1000) - start_time) / 1000
                    num_runtimes += 1
                    avg_runtime = (avg_runtime * (num_runtimes - 1) + runtime) / num_runtimes
                    runtime_string = get_time_string(runtime)
                    eta = get_time_string((iterations - iteration) * avg_runtime)
                    progress_row = '%d/%d\tseed=%d/%d \tinner_seed=%d/%d \tuser=%d/%d \ttime=%s \tETA=%s' % (
                        iteration, iterations, seed_idx + 1, len(seeds), inner_seed_idx + 1, len(inner_seeds),
                        user_count, num_users_to_test, runtime_string, eta)
                    print(progress_row)

    # average over inner folds
    df = pd.read_csv('%s/%s.csv' % (log_dir, file_name)).drop(columns='inner_seed')
    users = pd.unique(df['user'])
    groups_by_user = df.groupby('user')
    df_mean = None
    for user in users:
        df_user = groups_by_user.get_group(user)
        df_mean_by_seed = df_user.groupby('seed').mean()
        if df_mean is None:
            df_mean = df_mean_by_seed
        else:
            df_mean = df_mean.append(df_mean_by_seed)
    df_mean.insert(loc=1, column='seed', value=df_mean.index)
    df_mean.to_csv('%s/%s_by_seed.csv' % (log_dir, file_name), index=False)


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


# tree.plot_tree(baseline, feature_names=x_cols)
# plt.show()
