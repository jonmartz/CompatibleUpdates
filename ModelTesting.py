import csv
import os.path
import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
import Models
from ExperimentChooser import get_experiment_parameters


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def min_and_max(x):
    return pd.Series(index=['min', 'max'], data=[x.min(), x.max()])


def get_time_string(time_in_seconds):
    eta_string = '%.1f(secs)' % (time_in_seconds % 60)
    if time_in_seconds >= 60:
        time_in_seconds /= 60
        eta_string = '%d(mins) %s' % (time_in_seconds % 60, eta_string)
        if time_in_seconds >= 60:
            time_in_seconds /= 60
            eta_string = '%d(hours) %s' % (time_in_seconds % 24, eta_string)
            if time_in_seconds >= 24:
                time_in_seconds /= 24
                eta_string = '%d(days) %s' % (time_in_seconds, eta_string)
    return eta_string


if __name__ == "__main__":

    machine = 'PC'
    # machine = 'LIGHTSAIL'
    # machine = 'BGU-VPN'

    dataset_name = 'assistment'
    # dataset_name = 'ednet'
    # dataset_name = 'salaries'
    # dataset_name = 'recividism'

    # experiment settings
    chrono_split = False
    balance_histories = False

    # output settings
    make_tradeoff_plots = False
    show_tradeoff_plots = False
    plot_confusion = False

    target_col, original_categ_cols, user_cols, skip_cols, hists_already_determined, df_max_size, train_frac, valid_frac, \
    h1_len, h2_len, seeds, inner_seeds, weights_num, weights_range, model_type, max_depth, ccp_alpha, ridge_alpha, \
    min_hist_len, max_hist_len, metrics = get_experiment_parameters(dataset_name)
    if machine == 'PC':
        dataset_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/%s' % dataset_name
        result_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/current result'
    elif machine == 'LIGHTSAIL':
        dataset_dir = '/home/ubuntu/datasets/%s' % dataset_name
        result_dir = '/home/ubuntu/results'
    elif machine == 'BGU-VPN':
        dataset_dir = '/home/local/BGU-USERS/martijon/datasets/%s' % dataset_name
        result_dir = '/home/local/BGU-USERS/martijon/results'

    # model settings
    models_to_test = {
        'no hist': [1, 1, 0, 0],
        'L1': [0, 0, 1, 1],
        'L2': [0, 1, 1, 0],
        'L3': [0, 1, 1, 1],
        'L4': [1, 0, 0, 1],
        'L5': [1, 0, 1, 1],
        'L6': [1, 1, 0, 1],
        'L7': [1, 1, 1, 0],
        'L8': [1, 1, 1, 1],
    }

    # default settings
    diss_weights = np.array([i / weights_num for i in range(weights_num + 1)])
    diss_weights = (diss_weights * (weights_range[1] - weights_range[0]) + weights_range[0]).tolist()
    model_names = list(models_to_test.keys())

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

    # create dataset dir
    dataset_path = '%s/%s.csv' % (dataset_dir, dataset_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        with open('%s/parameters.csv' % result_dir, 'w', newline='') as file_out:
            writer = csv.writer(file_out)
            writer.writerow(['train_frac', 'valid_frac', 'ccp_alpha', 'dataset_max_size', 'h1_len', 'h2_len', 'seeds',
                             'inner_seeds', 'weights_num', 'weights_range', 'min_hist_len', 'max_hist_len', 'chrono_split',
                             'balance_histories', 'skip_cols', 'model_type'])
            writer.writerow([train_frac, valid_frac, ccp_alpha, df_max_size, h1_len, h2_len, len(seeds), len(inner_seeds),
                             weights_num, str(weights_range), min_hist_len, max_hist_len, chrono_split, balance_histories,
                             str(skip_cols), model_type])

    # run whole experiment for each user column selection
    for user_col in user_cols:
        print('user column = %s' % user_col)
        done_by_seed = {}

        # create all folders
        result_type_dir = '%s/%s' % (result_dir, user_col)
        if not os.path.exists(result_type_dir):
            for metric in metrics:
                os.makedirs('%s/%s' % (result_type_dir, metric))
            header = ['user', 'len', 'seed', 'inner_seed', 'h1_acc', 'weight']
            for model_name in model_names:
                header.extend(['%s x' % model_name, '%s y' % model_name])
            for metric in metrics:
                for subset in ['train', 'valid', 'test']:
                    with open('%s/%s/%s_log.csv' % (result_type_dir, metric, subset), 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(header)
            if make_tradeoff_plots:
                os.makedirs('%s/plots' % result_type_dir)

        else:  # load what is already done
            done_by_seed = {}
            df_done = pd.read_csv('%s/%s/test_log.csv' % (result_type_dir, metrics[-1]))
            groups_by_seed = df_done.groupby('seed')
            for seed, seed_group in groups_by_seed:
                done_by_inner_seed = {}
                done_by_seed[seed] = done_by_inner_seed
                groups_by_inner_seed = seed_group.groupby('inner_seed')
                for inner_seed, inner_seed_group in groups_by_inner_seed:
                    done_by_inner_seed[inner_seed] = len(pd.unique(inner_seed_group['user']))
            del df_done

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

            print('one-hot encoding the data... ')
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
            dataset_full = ce.OneHotEncoder(cols=categ_cols, use_cat_names=True).fit_transform(dataset_full)

            if hists_already_determined:  # todo: handle multiple seeds when balancing
                dataset_full.to_csv('%s/0.csv' % cache_dir, index=False)
                if not os.path.exists('%s/all_columns.csv' % cache_dir):
                    pd.DataFrame(columns=list(dataset_full.drop(columns=[user_col]).columns)).to_csv(
                        '%s/all_columns.csv' % cache_dir, index=False)
                del dataset_full
            else:
                print('sorting histories...')
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
                                hist = target_groups.apply(
                                    lambda x: x.sample(target_groups.size().min(), random_state=seed))
                                hist.index = hist.index.droplevel(0)
                            hists[user_id] = hist
                        sorted_hists = [[k, v] for k, v in reversed(sorted(hists.items(), key=lambda n: len(n[1])))]
                        seed_df = pd.DataFrame(columns=[user_col] + all_columns, dtype=np.int64)
                        for user_id, hist in sorted_hists:
                            hist[user_col] = [user_id] * len(hist)
                            seed_df = seed_df.append(hist, sort=False)
                        seed_df.to_csv('%s/%d.csv' % (cache_dir, seed), index=False)
                    if not balance_histories:
                        break
                del groups_by_user
                del hists
        # end of making seed caches

        print("determine experiment's users...")
        min_max_col_values = pd.read_csv('%s/all_columns.csv' % cache_dir, dtype=np.int64)
        all_columns = min_max_col_values.columns
        groups_by_user = pd.read_csv('%s/0.csv' % cache_dir).groupby(user_col, sort=False)
        hists_by_user = {}
        hist_train_ranges = {}
        curr_h2_len = 0
        for user_id, hist in groups_by_user:
            hist = hist.drop(columns=[user_col])
            if hists_already_determined or (min_hist_len <= len(hist) * train_frac
                                            and curr_h2_len + len(hist) * train_frac <= h2_len):
                if len(hist) > max_hist_len:
                    hist = hist[:max_hist_len]
                hist_train_ranges[user_id] = [curr_h2_len, curr_h2_len + int(len(hist) * train_frac)]
                curr_h2_len += int(len(hist) * train_frac)
                hists_by_user[user_id] = hist
                min_max_col_values = min_max_col_values.append(hist.apply(min_and_max), sort=False)
                if curr_h2_len + min_hist_len * train_frac > h2_len:
                    break
        del groups_by_user

        # set hist train ranges
        for user_id, hist_train_range in hist_train_ranges.items():
            hist_train_len = hist_train_range[1] - hist_train_range[0]
            range_vector = np.zeros(curr_h2_len)
            for i in range(hist_train_range[0], hist_train_range[1]):
                range_vector[i] = 1
            hist_train_ranges[user_id] = [range_vector, hist_train_len]

        print('cols=%d train_len=%d diss_weights=%d users=%d model_type=%s' % (
            len(all_columns) - 1, curr_h2_len, len(diss_weights), len(hists_by_user), model_type))

        min_max_col_values = min_max_col_values.reset_index(drop=True)
        scaler, labelizer = MinMaxScaler(), LabelBinarizer()
        scaler.fit(min_max_col_values.drop(columns=[target_col]), min_max_col_values[[target_col]])
        labelizer.fit(min_max_col_values[[target_col]])
        del min_max_col_values

        print('\nstart experiment!')
        iterations = len(seeds) * len(inner_seeds) * len(hists_by_user)
        iteration = 0
        avg_runtime = 0
        num_runtimes = 0
        for seed_idx, seed in enumerate(seeds):

            if seed in done_by_seed:  # check if seed was already done
                done_by_inner_seed = done_by_seed[seed]
                seed_is_done = len(done_by_inner_seed) == len(inner_seeds) and all(
                    [done_users == len(hists_by_user) for i, done_users in done_by_inner_seed.items()])
            else:
                done_by_inner_seed = {}
                seed_is_done = False
            if seed_is_done:
                iteration += len(inner_seeds) * len(hists_by_user)
                continue

            # split the test sets
            hists_seed_by_user = {}
            for user_id, hist in hists_by_user.items():
                hist_train_and_valid = hist.sample(n=int(len(hist) * (train_frac + valid_frac)), random_state=seed)
                hist_test = hist.drop(hist_train_and_valid.index).reset_index(drop=True)
                hist_test_x = scaler.transform(hist_test.drop(columns=[target_col]))
                hist_test_y = labelizer.transform(hist_test[[target_col]])
                hists_seed_by_user[user_id] = [hist_train_and_valid, hist_test_x, hist_test_y]

            # run experiment for each inner seed
            for inner_seed_idx, inner_seed in enumerate(inner_seeds):

                if inner_seed in done_by_inner_seed:  # check if inner seed was already done
                    done_last_users = done_by_inner_seed[inner_seed]
                    inner_seed_is_done = done_last_users == len(hists_by_user)
                else:
                    done_last_users = 0
                    inner_seed_is_done = False
                if inner_seed_is_done:
                    iteration += len(hists_by_user)
                    continue

                # split to train and validation sets
                hists_inner_seed_by_user = {}
                h2_train = pd.DataFrame(columns=all_columns, dtype=np.int64)
                for user_id, item in hists_seed_by_user.items():
                    hist_train_and_valid, hist_test_x, hist_test_y = item
                    hist_train_len = hist_train_ranges[user_id][1]
                    hist_train = hist_train_and_valid.sample(n=hist_train_len, random_state=inner_seed)
                    hist_valid = hist_train_and_valid.drop(hist_train.index)
                    h2_train = h2_train.append(hist_train, ignore_index=True, sort=False)
                    hists_inner_seed_by_user[user_id] = [hist_train, hist_valid, hist_test_x, hist_test_y]
                h2_train_x = scaler.transform(h2_train.drop(columns=[target_col]))
                h2_train_y = labelizer.transform(h2_train[[target_col]])

                # train h1 and baseline
                h1 = Models.get_model(model_type, h2_train_x[:h1_len], h2_train_y[:h1_len], model_name='h1',
                                      ccp_alpha=ccp_alpha, max_depth=max_depth, ridge_alpha=ridge_alpha)
                no_hists = []
                for weight in diss_weights:
                    no_hists.append(
                        Models.get_model(model_type, h2_train_x, h2_train_y, subset_weights=[1, 1, 0, 0], old_model=h1,
                                         diss_weight=weight, ccp_alpha=ccp_alpha, max_depth=max_depth,
                                         ridge_alpha=ridge_alpha))
                # run experiment for each user in inner seed
                user_count = 0
                for user_id, item in hists_inner_seed_by_user.items():
                    iteration += 1
                    user_count += 1
                    if user_count <= done_last_users:
                        continue
                    start_time = int(round(time() * 1000))

                    # prepare train and validation sets
                    hist_train, hist_valid, hist_test_x, hist_test_y = item
                    hist_train_range = hist_train_ranges[user_id][0]
                    hist_len = len(hist_train) + len(hist_valid) + len(hist_test)
                    hist_train_x = scaler.transform(hist_train.drop(columns=[target_col]))
                    hist_train_y = labelizer.transform(hist_train[[target_col]])
                    hist_valid_x = scaler.transform(hist_valid.drop(columns=[target_col]))
                    hist_valid_y = labelizer.transform(hist_valid[[target_col]])

                    # train all models
                    models_by_weight = []
                    for weight_idx, weight in enumerate(diss_weights):
                        models = []
                        models_by_weight.append(models)
                        for model_name in model_names:
                            if model_name == 'no hist':
                                model = no_hists[weight_idx]
                            else:
                                subset_weights = models_to_test[model_name]
                                model = Models.get_model(
                                    model_type, h2_train_x, h2_train_y, subset_weights=subset_weights, old_model=h1,
                                    diss_weight=weight, hist_range=hist_train_range, ccp_alpha=ccp_alpha,
                                    max_depth=max_depth, ridge_alpha=ridge_alpha)
                            models.append(model)

                    # test all models
                    rows_by_metric = []
                    for metric in metrics:
                        rows_by_subset = []
                        rows_by_metric.append(rows_by_subset)
                        for subset in ['train', 'valid', 'test']:
                            x, y = eval('hist_%s_x' % subset), eval('hist_%s_y' % subset)
                            rows = []
                            rows_by_subset.append(rows)
                            h1_y = h1.test(x, y, metric)['y']
                            for weight_idx, weight in enumerate(diss_weights):
                                models = models_by_weight[weight_idx]
                                row = [user_id, hist_len, seed, inner_seed, h1_y, weight]
                                for model in models:
                                    result = model.test(x, y, metric)
                                    row.extend([result['x'], result['y']])
                                rows.append(row)

                    # write rows to all logs in one go to avoid discrepancies between logs
                    for metric_idx, metric in enumerate(metrics):
                        for subset_idx, subset in enumerate(['train', 'valid', 'test']):
                            with open('%s/%s/%s_log.csv' % (result_type_dir, metric, subset), 'a', newline='') as file:
                                writer = csv.writer(file)
                                for row in rows_by_metric[metric_idx][subset_idx]:
                                    writer.writerow(row)

                    # end iteration
                    runtime = (round(time() * 1000) - start_time) / 1000
                    num_runtimes += 1
                    avg_runtime = (avg_runtime * (num_runtimes - 1) + runtime) / num_runtimes
                    runtime_string = get_time_string(runtime)
                    eta = get_time_string((iterations - iteration) * avg_runtime)
                    progress_row = '%d/%d\tseed=%d/%d \tinner_seed=%d/%d \tuser=%d/%d \ttime=%s \tETA=%s' % \
                                   (iteration, iterations, seed_idx + 1, len(seeds), inner_seed_idx + 1, len(inner_seeds),
                                    user_count, len(hists_by_user), runtime_string, eta)
                    with open('%s/progress_log.txt' % result_type_dir, 'a') as file:
                        file.write('%s\n' % progress_row)
                    print(progress_row)
                # end user loop
            # end inner seed loop
        # end seed loop
    # end user type loop

    print('\ndone')
