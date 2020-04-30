import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_users(log_dir, log_set):
    df_log = pd.read_csv('%s\\%s_log.csv' % (log_dir, log_set))
    log_by_users = df_log.groupby('user')

    for user_id, user_data in log_by_users:
        try:
            mod_user_id = []
            for row in user_data['seed']:
                mod_user_id += [str(user_id) + '_' + str(row)]
            user_data['user'] = mod_user_id
        except KeyError:
            user_data['user'] = user_id
        user_data.to_csv('%s\\users_%s\\logs\\%s_%s.csv' % (log_dir, log_set, log_set, user_id), index=False)


def get_model_dict(cmap_name):
    models = {
        'no diss': {'sample_weight': [1, 0, 1, 0], 'color': 'grey'},
        'no hist': {'sample_weight': [1, 1, 0, 0], 'color': 'black'},
        # 'sim_ann': {'sample_weight': [0.0, 0.6352, 0.3119, 0.0780], 'color': 'purple'},
        'hybrid': {'sample_weight': ['', '', '', ''], 'color': 'green'},
        'best_u': {'sample_weight': ['', '', '', ''], 'color': 'red'},
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
    cmap = plt.cm.get_cmap(cmap_name)
    for i in range(len(parametrized_models)):
        model = parametrized_models[i]
        models[model[0]] = {'sample_weight': model[1], 'color': cmap((i + 1) / (len(parametrized_models) + 3))}
    return models


def plot_results(log_dir, dataset, user_type, models, log_set, bin_size=1, user_name='', show_tradeoff_plots=False,
                 smooth_color_progression=False):
    if user_name == '':
        df_results = pd.read_csv('%s\\%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s\\logs\\%s_%s.csv' % (log_dir, log_set, user_name))

    # skip_users = [
    #     # 'Husband',
    #     # 'Not-in-family',
    #     # 'Own-child',
    # ]
    # df_results = df_results[~df_results['user'].isin(skip_users)]

    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    xs, ys, xs_plot, ys_plot = [], [], [], []
    autcs = []
    h1_avg_acc = np.average(df_results['h1_acc'], weights=df_results['len'])
    weights = pd.unique(df_results['weight'])
    groups_by_weight = df_results.groupby('weight')

    fig, (ax, tabax) = plt.subplots(nrows=2, figsize=(6.4, 4.8 + 0.3 * len(model_names)))
    cmap = plt.cm.get_cmap('jet')

    dfs_by_weight = [groups_by_weight.get_group(i) for i in weights]
    for model_name in model_names:
        if model_name not in models.keys():
            continue
        x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
        y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]

        # make x monotonic for AUTC
        # x_monotonic = x.copy()
        # for i in range(1, len(x_monotonic)):
        #     if x_monotonic[i] < x_monotonic[i - 1]:
        #         x_monotonic[i] = x_monotonic[i - 1]
        # h1_area = (x_monotonic[-1] - x_monotonic[0]) * h1_avg_acc
        # autc = auc(x_monotonic, y) - h1_area

        for i in range(1, len(x)):
            if x[i] < x[i - 1]:
                x[i] = x[i - 1]

        h1_area = (x[-1] - x[0]) * h1_avg_acc
        autc = auc(x, y) - h1_area

        autcs.append(autc)
        if model_name == 'no hist':
            no_hist_autc = autc

        xs.append(x)
        ys.append(y)
        if bin_size > 1:  # bin points
            last_idx = len(x) - ((len(x) - 2) % bin_size) - 1
            x_binned = np.mean(np.array(x[1:last_idx]).reshape(-1, bin_size), axis=1)
            y_binned = np.mean(np.array(y[1:last_idx]).reshape(-1, bin_size), axis=1)
            xs_plot.append([x[0]] + list(x_binned) + [np.mean(x[last_idx:-1]), x[-1]])
            ys_plot.append([y[0]] + list(y_binned) + [np.mean(y[last_idx:-1]), y[-1]])
        else:
            xs_plot.append(x)
            ys_plot.append(y)

    min_x = min(min(i) for i in xs)
    h1_x = [min_x, max(max(i) for i in xs)]
    h1_y = [h1_avg_acc, h1_avg_acc]
    ax.plot(h1_x, h1_y, 'k--', marker='.', label='h1')

    # adding area on left to no hist for correct autc improvement calculation
    x_no_hist_ = xs[0]
    y_no_hist_ = ys[0]
    min_x_no_hist_ = min(x_no_hist_)
    if min_x_no_hist_ > min_x:  # for models that start at better compatibility
        no_hist_autc += (min_x_no_hist_ - min_x) * (y_no_hist_[0] - h1_avg_acc)

    autc_improvs = []
    best_model = ''
    best_autc = None
    for i in range(len(model_names)):
        model_name = model_names[i]
        if model_name not in models.keys():
            continue
        x = xs[i]
        y = ys[i]
        autc = autcs[i]
        min_x_model = min(x)
        if min_x_model > min_x:  # for models that start at better compatibility
            autc += (min_x_model - min_x) * (y[0] - h1_avg_acc)
        if best_autc is None or autc > best_autc:
            best_autc = autc
            best_model = model_names[i]
        autc_improvs.append((autc / no_hist_autc - 1) * 100)

    sorted_idxs = [idx for autc, idx in reversed(sorted(zip(autc_improvs, range(len(autc_improvs)))))]

    cell_text = []
    model_names_sorted = []
    colors = []

    color_idx = 1
    for i in sorted_idxs:
        x_plot, y_plot = xs_plot[i], ys_plot[i]
        autc_improv = autc_improvs[i]
        if autc_improv >= 0:
            sign = '+'
        else:
            sign = ''
        model_name = model_names[i]
        model_names_sorted.append(model_name)
        model = models[model_name]
        if smooth_color_progression:
            if model_name == 'no hist':
                color = 'black'
            else:
                color = cmap(color_idx / (len(model_names) + 1))
                color_idx += 1
        else:
            color = model['color']

        sample_weight = model['sample_weight']
        if model_name == 'sim_ann':
            sample_weight = ['%.3f' % i for i in sample_weight]
        cell_text += [sample_weight + ['%s%.1f%%' % (sign, autc_improv)]]
        if model_name == 'best_u':
            x_best, y_best, best_color = x_plot, y_plot, color
        else:
            ax.plot(x_plot, y_plot, label=model_name, color=color)
        if model_name == 'no hist':
            color = 'white'
        colors.append(color)
    if 'best_u' in model_names:
        ax.plot(x_best, y_best, label='best_u', color=best_color)

    # table
    columns = ('gen', 'gen diss', 'hist', 'hist diss', '+ AUTC %')
    tabax.axis("off")
    tabax.table(cellText=cell_text, rowLabels=model_names_sorted, rowColours=colors, colLabels=columns, loc='center')

    ax.set_xlabel('compatibility')
    ax.set_ylabel('accuracy')
    if user_name == '':
        title = 'dataset=%s user_type=%s' % (dataset, user_type)
        save_name = '%s\\%s_plots.png' % (log_dir, log_set)
    else:
        len_h = df_results.loc[0]['len']
        title = 'dataset=%s user=%s len(h)=%d' % (dataset, user_name, len_h)
        save_name = '%s\\%s_%s.png' % (log_dir, log_set, user_name)
    ax.set_title(title)
    plt.savefig(save_name, bbox_inches='tight')
    if show_tradeoff_plots:
        plt.show()
    plt.clf()

    return best_model
    # todo: return best model by weight


def get_best_models(log_dir, models, log_set, user_name='', plot_tradeoffs=False):
    if user_name == '':
        df_results = pd.read_csv('%s\\%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s\\%s_%s.csv' % (log_dir, log_set, user_name))

    if plot_tradeoffs:
        seed_plots_dir = '%s\\%s_seed_plots' % (log_dir, log_set)
        safe_make_dir(seed_plots_dir)

    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    seeds = pd.unique(df_results['seed']).tolist()
    groups_by_seed = df_results.groupby('seed')
    weights = pd.unique(df_results['weight'])
    best_models_by_seed = []

    for seed_idx in range(len(seeds)):
        seed = seeds[seed_idx]
        print('\t%d/%d seed %d' % (seed_idx + 1, len(seeds), seed))
        df_seed = groups_by_seed.get_group(seed)
        h1_avg_acc = np.average(df_seed['h1_acc'], weights=df_seed['len'])
        groups_by_weight = df_seed.groupby('weight')
        if user_name == '':
            dfs_by_weight = [groups_by_weight.get_group(i) for i in weights]
        else:
            means = groups_by_weight.mean()

        autcs = []
        xs_seed, ys_seed = [], []
        for model_name in model_names:
            if model_name not in models.keys():
                continue
            if user_name == '':
                x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
                y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]
            else:
                x = means['%s x' % model_name].tolist()
                y = means['%s y' % model_name].tolist()

            for i in range(1, len(x)):
                if x[i] < x[i - 1]:
                    x[i] = x[i - 1]
            h1_area = (x[-1] - x[0]) * h1_avg_acc
            autc = auc(x, y) - h1_area
            autcs.append(autc)
            xs_seed.append(x)
            ys_seed.append(y)
        min_x = min(min(i) for i in xs_seed)
        max_x = max(max(i) for i in xs_seed)

        # get best model by seed
        best_model = ''
        best_autc = None
        for i in range(len(model_names)):
            model_name = model_names[i]
            if model_name not in models.keys():
                continue
            x = [min_x] + xs_seed[i]
            y = ys_seed[i]
            y = [y[0]] + y
            autc = autcs[i]
            if best_autc is None or autc > best_autc:
                best_autc = autc
                best_model = model_name
            if plot_tradeoffs:
                plt.plot(x, y, label='%s autc=%.5f' % (model_name, autc))
        if plot_tradeoffs:
            plt.plot([min_x, max_x], [h1_avg_acc, h1_avg_acc], 'k--', label='h1')
            plt.xlabel('compatibility')
            plt.ylabel('accuracy')
            plt.legend()
            plt.title('user=%s seed=%d best=%s' % (user_name, seed, best_model))
            plt.savefig('%s\\user_%s seed_%d' % (seed_plots_dir, user_name, seed))
            plt.clf()
        best_models_by_seed.append(best_model)
    return seeds, best_models_by_seed
    # todo: return best model by weight


def add_best_model(log_dir, valid_set='valid'):
    # # GET BEST MODELS FROM VALIDATION SET #
    # df_valid = pd.read_csv('%s\\valid_log.csv' % log_dir)
    # model_names = [i[:-2] for i in df_valid.columns if ' x' in i]
    # seeds = pd.unique(df_valid['seed'])
    #
    # # # getting best model by compatibility value
    # # groups_by_weight = df_valid.groupby('weight')
    # # dfs_by_weight = [groups_by_weight.get_group(i) for i in groups_by_weight.groups]
    # # best_by_comp = {}
    # # best_accuracy_by_comp = None
    # # for model_name in model_names:
    # #     x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
    # #     y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]
    #
    # # getting best model by user
    # groups_by_user = df_valid.groupby('user')
    # user_names = pd.unique(df_valid['user'])
    # best_model_by_user = {}
    # best_by_comp_by_user = {}
    # for user_name in user_names:
    #     # todo: implement by seed
    #     df_user = groups_by_user.get_group(user_name)
    #     # groups_by_weight = df_user.groupby('weight')
    #     # dfs_by_weight = [groups_by_weight.get_group(i) for i in groups_by_weight.groups]
    #     means_by_weight = df_user.groupby('weight').mean()
    #
    #     # compute raw autcs
    #     first_x, first_y, autcs = [], [], []
    #     for model_name in model_names:
    #         # x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
    #         # y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]
    #         x = means_by_weight['%s x' % model_name].tolist()
    #         y = means_by_weight['%s y' % model_name].tolist()
    #         for i in range(1, len(x)):  # make monotonic
    #             x[i] = max(x[i], x[i - 1])
    #         first_x.append(x[0])
    #         first_y.append(y[0])
    #         autcs.append(auc(x, y))
    #
    #     # extend plots to min_x of all models' x
    #     best_autc = 0
    #     min_x = min(first_x)
    #     for i in range(len(model_names)):
    #         model_name = model_names[i]
    #         autc = autcs[i] + first_y[i] * (first_x[i] - min_x)
    #         if autc > best_autc:
    #             best_model_by_user[user_name] = model_name
    #             best_autc = autc

    # NEW

    # df_test = pd.read_csv('%s\\test_log.csv' % log_dir)
    # user_names = pd.unique(df_test['user'])
    # groups_by_user = df_test.groupby('user')
    # best_models = pd.read_csv('%s\\best_models_%s.csv' % (log_dir, valid_set))['model']
    # new_model_x = []
    # new_model_y = []
    # for i in range(len(user_names)):
    #     user_name = user_names[i]
    #     df_user = groups_by_user.get_group(user_name)
    #     best_model = best_models[i]
    #     new_model_x += df_user['%s x' % best_model].tolist()
    #     new_model_y += df_user['%s y' % best_model].tolist()
    # df_test['best_u x'] = new_model_x
    # df_test['best_u y'] = new_model_y
    # df_test.to_csv('%s\\test_with_best_log.csv' % log_dir, index=False)

    df_test = pd.read_csv('%s\\test_log.csv' % log_dir)
    groups_by_user = df_test.groupby('user')
    df_best = pd.read_csv('%s\\best_models_%s.csv' % (log_dir, valid_set))
    user_names = pd.unique(df_best['user'])
    seeds = pd.unique(df_best['seed'])
    best_models = df_best.to_numpy()
    new_model_x = []
    new_model_y = []
    i = 0
    for user_name in user_names:
        df_user = groups_by_user.get_group(user_name)
        groups_by_seed = df_user.groupby('seed')
        for seed in seeds:
            df_seed = groups_by_seed.get_group(seed)
            best_user, best_seed, best_model = best_models[i]
            new_model_x += df_seed['%s x' % best_model].tolist()
            new_model_y += df_seed['%s y' % best_model].tolist()
            # print('(%s=%s) (%d=%d)' % (user_name, best_user, seed, best_seed))
            if user_name != best_user or seed != best_seed:
                raise ValueError('results and best lists not in same order of user -> seed')
            i += 1

    df_test['best_u x'] = new_model_x
    df_test['best_u y'] = new_model_y
    df_test.to_csv('%s\\test_with_best_log.csv' % log_dir, index=False)


def binarize_results_by_compat_values(log_dir, log_set, bin_num=100):
    df_results = pd.read_csv('%s\\%s_log.csv' % (log_dir, log_set))
    user_names = pd.unique(df_results['user'])
    seeds = pd.unique(df_results['seed'])
    groups_by_user = df_results.groupby('user')
    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    df_bins = pd.DataFrame(columns=df_results.columns, dtype=np.int64)
    bins = np.array([i / bin_num for i in range(bin_num + 1)])
    weights = np.array(range(bin_num + 1)) / bin_num

    for user_idx in range(len(user_names)):
        user_name = user_names[user_idx]
        print('%d/%d user=%d' % (user_idx + 1, len(user_names), user_name))
        df_user = groups_by_user.get_group(user_name)
        df_bins_user = pd.DataFrame(columns=df_results.columns, dtype=np.int64)
        groups_by_seed = df_user.groupby('seed')

        for seed_idx in seeds:
            seed = seeds[seed_idx]
            print('\t%d/%d seed=%d' % (seed_idx + 1, len(seeds), seed))
            df_seed = groups_by_seed.get_group(seed)
            df_by_weight = df_seed.groupby('weight').mean()
            h1_y = df_by_weight['h1_acc'].iloc[0]
            df_bins_user_seed = pd.DataFrame({'user': user_name, 'len': df_user['len'].iloc[0],
                                              'h1_acc': h1_y, 'weight': weights})
            xs = []
            ys = []
            for model_name in model_names:
                x = df_by_weight['%s x' % model_name].tolist()
                y = df_by_weight['%s y' % model_name].tolist()
                for i in range(1, len(x)):  # make monotonic increasing
                    if x[i] < x[i - 1]:
                        x[i] = x[i - 1]
                xs.append(x)
                ys.append(y)

            # add min and max x to each model
            min_x = min([min(i) for i in xs])
            max_x = max([max(i) for i in xs])
            for i in range(len(model_names)):
                x = xs[i]
                y = ys[i]
                xs[i] = [min_x] + x + [x[-1], max_x]
                ys[i] = [y[0]] + y + [h1_y, h1_y]

            # binarize
            for idx in range(len(model_names)):
                model_name = model_names[idx]
                x = xs[idx]
                y = ys[idx]
                x_bins = (bins * (x[-1] - x[0]) + x[0]).tolist()
                y_bins = []
                i = 0
                for x_bin in x_bins:  # get y given x for each x_bin
                    while not x[i] <= x_bin <= x[i + 1]:
                        i += 1
                    x_left, x_right, y_left, y_right = x[i], x[i + 1], y[i], y[i + 1]
                    if x_left == x_right:  # vertical line
                        y_bins.append(max(y_left, y_right))
                    else:
                        slope = (y_right - y_left) / (x_right - x_left)
                        y_bins.append(y_left + slope * (x_bin - x_left))
                df_bins_user_seed['%s x' % model_name] = x_bins
                df_bins_user_seed['%s y' % model_name] = y_bins
            df_bins_user = df_bins_user.append(df_bins_user_seed, sort=True)
        df_bins = df_bins.append(df_bins_user, sort=True)
    df_bins.to_csv('%s\\%s_bins_log.csv' % (log_dir, log_set), index=False)

    # print('binarizing...')
    #     df_results = pd.read_csv('%s\\%s_log.csv' % (log_dir, log_set))
    #     user_names = pd.unique(df_results['user'])
    #     groups_by_user = df_results.groupby('user')
    #     seeds = pd.unique(df_results['seed'])
    #     model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    #     weights = np.array(range(bins + 1)) / bins
    #     df_bins = pd.DataFrame(columns=df_results.columns, dtype=np.int64)
    #     user_idx = 0
    #     for user_name in user_names:
    #         user_idx += 1
    #         print('%d/%d user = %s' % (user_idx, len(user_names), user_name))
    #         df_user = groups_by_user.get_group(user_name)
    #         groups_by_seed = df_user.groupby('seed')
    #         for seed in seeds:
    #             df_user_seed = groups_by_seed.get_group(seed)
    #             df_by_weight = df_user_seed.groupby('weight').mean()
    #             df_user_seed_bins = pd.DataFrame({'user': user_name, 'len': df_user_seed['len'].iloc[0], 'seed': seed,
    #                                          'h1_acc': df_by_weight['h1_acc'].iloc[0], 'weight': weights})
    #             for model_name in model_names:
    #                 x = df_by_weight['%s x' % model_name].tolist()
    #                 y = df_by_weight['%s y' % model_name].tolist()
    #                 for i in range(1, len(x)):
    #                     if x[i] < x[i - 1]:
    #                         x[i] = x[i - 1]
    #                 x_bins = np.array([i / bins for i in range(1, bins)])
    #                 x_bins = (x_bins * (x[-1] - x[0]) + x[0]).tolist()
    #                 y_bins = []
    #                 i = 0
    #                 for x_bin in x_bins:  # get y given x for each x_bin
    #                     while not x[i] <= x_bin <= x[i + 1]:
    #                         i += 1
    #                     x_left, x_right, y_left, y_right = x[i], x[i + 1], y[i], y[i + 1]
    #                     if x_left == x_right:  # vertical line
    #                         y_bins.append(max(y_left, y_right))
    #                     else:
    #                         slope = (y_right - y_left) / (x_right - x_left)
    #                         y_bins.append(y_left + slope * (x_bin - x_left))
    #                 df_user_seed_bins['%s x' % model_name] = [x[0]] + x_bins + [x[-1]]
    #                 df_user_seed_bins['%s y' % model_name] = [y[0]] + y_bins + [y[-1]]
    #             df_bins = df_bins.append(df_user_seed_bins)
    #     df_bins.to_csv('%s\\%s_bins_log.csv' % (log_dir, log_set), index=False)


def best_count_values(log_dir, log_set):
    df_best = pd.read_csv('%s\\best_models_%s.csv' % (log_dir, log_set))
    groups_by_users = df_best.groupby('user')
    models = pd.unique(df_best['model'])
    users = pd.unique(df_best['user'])
    dict_counts = {i: [] for i in models}
    dict_counts['user'] = users
    for user in users:
        df_user = groups_by_users.get_group(user)
        value_counts = df_user['model'].value_counts()
        for model in models:
            try:
                dict_counts[model].append(0)
            except AttributeError:
                print('hi')
        for model in list(value_counts.index):
            dict_counts[model][-1] = dict_counts[model][-1] + value_counts[model]
    df_result = pd.DataFrame(dict_counts)
    df_result.to_csv('%s\\best_models_%s_counts.csv' % (log_dir, log_set), index=False)


dataset = 'assistment'
version = 'unbalanced\\without skills'
user_type = 'user_id'

# dataset = 'salaries'
# version = 'unbalanced\\1'
# user_type = 'relationship'

# log_set = 'test'
log_set = 'valid'
# log_set += '_with_best'
log_set += '_bins'
individual_users = True
get_best = False
add_best = False
count_best = False
binarize_by_compat = False

bin_size = 2

# steps to get average tradeoff:
# 1. valid + individual_users + get_best
# 2. add_best
# 3. test_with_best + binarize_by_compat
# 4. test_with_best_bins

results_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\simulated annealing'
log_dir = '%s\\%s\\%s\\%s' % (results_dir, dataset, version, user_type)
models = get_model_dict('jet')
# models = get_model_dict('gist_rainbow')

if add_best:
    add_best_model(log_dir)
elif count_best:
    best_count_values(log_dir, log_set)
elif binarize_by_compat:
    binarize_results_by_compat_values(log_dir, log_set)
elif not individual_users:
    if get_best:
        # best_model = get_best_models(log_dir, models, log_set)
        print('got best models for general set, not individual users!')
    else:
        plot_results(log_dir, dataset, user_type, models, log_set, bin_size=bin_size, show_tradeoff_plots=True)
else:
    users_dir = '%s\\users_%s' % (log_dir, log_set)
    user_logs_dir = '%s\\logs' % users_dir
    if not os.path.exists(users_dir):
        safe_make_dir(user_logs_dir)
        split_users(log_dir, log_set)
    user_ids = pd.unique(pd.read_csv('%s\\%s_log.csv' % (log_dir, log_set))['user'])
    if get_best:
        df = pd.DataFrame(columns=['user', 'seed', 'model'])
        user_col = []
        seed_col = []
        model_col = []
        for user_idx in range(len(user_ids)):
            user_id = user_ids[user_idx]
            print('%d/%d seed %d' % (user_idx + 1, len(user_ids), user_id))
            seeds, best_models_by_seed = get_best_models('%s\\users_%s\\logs' % (log_dir, log_set), models, log_set,
                                                         user_name=user_id)
            user_col += [user_id] * len(seeds)
            seed_col += seeds
            model_col += best_models_by_seed
        df = pd.DataFrame({'user': user_col, 'seed': seed_col, 'model': model_col})
        df.to_csv('%s\\best_models_%s.csv' % (log_dir, log_set), index=False)
    else:
        for user_idx in range(len(user_ids)):
            user_id = user_ids[user_idx]
            print('%d/%d user=%s' % (user_idx + 1, len(user_ids), user_id))
            plot_results('%s\\users_%s' % (log_dir, log_set), dataset, user_type, models, log_set,
                         bin_size=bin_size, user_name=user_id)
