import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from postprocessing.DataAnalysis import make_data_analysis, make_data_analysis_per_inner_seed
from scipy.stats import ttest_rel
from ExperimentChooser import get_experiment_parameters
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_users(log_dir, log_set):
    df_log = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    log_by_users = df_log.groupby('user')
    print('\nsplitting %s log into user logs...' % log_set)
    user_idx = 0
    for user_id, user_data in log_by_users:
        user_idx += 1
        print('\tuser %d/%d' % (user_idx, len(log_by_users)))
        try:
            mod_user_id = []
            for row in user_data['seed']:
                mod_user_id += [str(user_id) + '_' + str(row)]
            user_data['user'] = mod_user_id
        except KeyError:
            user_data['user'] = user_id
        user_data.to_csv('%s/users_%s/logs/%s_%s.csv' % (log_dir, log_set, log_set, user_id), index=False)


def get_model_dict(cmap_name):
    models = {
        # 'no diss': {'sample_weight': [1, 0, 1, 0], 'color': 'grey', 'std': False},
        'no hist': {'sample_weight': [1, 1, 0, 0], 'color': 'black', 'std': True},
        # 'sim_ann': {'sample_weight': [0.0, 0.6352, 0.3119, 0.0780], 'color': 'purple'},
        # 'hybrid': {'sample_weight': ['', '', '', ''], 'color': 'green', 'std': False},
        'best_u': {'sample_weight': ['', '', '', ''], 'color': 'red', 'std': True},
        'simple best_u': {'sample_weight': ['', '', '', ''], 'color': 'blue', 'std': True},
        # SYNTHETIC:
        # 'model1': {'sample_weight': ['', '', '', ''], 'color': 'red', 'std': True},
        # 'model2': {'sample_weight': ['', '', '', ''], 'color': 'blue', 'std': True},
    }
    if add_parametrized_models:
        parametrized_models = [  # [general_loss, general_diss, hist_loss, hist_diss]
            ['L1', [0, 0, 1, 1], False],
            ['L2', [0, 1, 1, 0], False],
            ['L3', [0, 1, 1, 1], False],
            ['L4', [1, 0, 0, 1], False],
            ['L5', [1, 0, 1, 1], False],
            ['L6', [1, 1, 0, 1], False],
            ['L7', [1, 1, 1, 0], False],
            ['L8', [1, 1, 1, 1], False],
        ]
        cmap = plt.cm.get_cmap(cmap_name)
        for i in range(len(parametrized_models)):
            model = parametrized_models[i]
            models[model[0]] = {'sample_weight': model[1], 'color': cmap((i + 1) / (len(parametrized_models) + 3)),
                                'std': model[2]}
    return models


# todo: this is by user
# def get_df_by_weight_norm(df, model_y_init_avg_acc, model_y_final_avg_acc, weights, model_names):
#     user_groups = df.groupby('user')
#     df_dict = {'len': [], 'weight': []}
#     entry_len = None
#     for model_name in model_names:
#         df_dict['%s y' % model_name] = []
#     for user, user_group in user_groups:
#         user_len = user_group['len'].iloc[0]
#         seed_groups = user_group.groupby('seed')
#         if entry_len is None:
#             entry_len = len(seed_groups) * len(weights)
#         df_dict['len'].extend([user_len] * entry_len)
#         for seed, seed_group in seed_groups:
#             df_dict['weight'].extend(weights)
#             seed_means = seed_group.groupby('weight').mean()
#             # seed_h1_avg = seed_means['h1_acc'].iloc[0]
#             for model_name in model_names:
#                 model_y = seed_means['%s y' % model_name]
#                 # y_min = min(min(model_y), seed_h1_avg)
#                 # y_max = max(max(model_y), seed_h1_avg)
#                 # if y_max == y_min:
#                 #     # if seed_h1_avg >= seed_no_hist_0_avg_acc or y_max == y_min:
#                 #     continue
#                 # model_y_norm = (model_y - model_y.iloc[0]) * ((no_hist_0_avg_acc[model_name] - h1_avg_acc) /
#                 #                                               (y_max - y_min)) + no_hist_0_avg_acc[model_name]
#                 # model_y_norm = (model_y - model_y.iloc[0]) * ((no_hist_init_avg_acc - no_hist_final_avg_acc) /
#                 #                                               (y_max - y_min)) + no_hist_init_avg_acc
#                 if max(model_y) == min(model_y):
#                     print('max(y) = min(y)')
#                 model_y_norm = (model_y - max(model_y)) * \
#                                ((model_y_init_avg_acc[model_name] - model_y_final_avg_acc[model_name]) /
#                                 (max(model_y) - min(model_y))) + model_y_init_avg_acc[model_name]
#                 df_dict['%s y' % model_name].extend(list(model_y_norm))
#
#     df_norm = pd.DataFrame(df_dict)
#     groups_by_weight = df_norm.groupby('weight')
#     return [groups_by_weight.get_group(i) for i in weights]


# todo: this is by seed
def get_df_by_weight_norm(df,
                          # model_y_init_avg_acc, model_y_final_avg_acc,
                          y_max_mean, y_min_mean,
                          weights, model_names):
    drop_columns = [i for i in df.columns if (' x' in i or (' y' in i and i[:-2] not in model_names))]
    drop_columns += ['user', 'inner_seed']
    df = df.drop(columns=drop_columns)
    df_dict = {'weight': []}
    for model_name in model_names:
        df_dict['%s y' % model_name] = []
    seed_groups = df.groupby('seed')
    for seed, seed_group in seed_groups:
        df_dict['weight'].extend(weights)
        weight_groups = seed_group.groupby('weight')
        models_y = {i: [] for i in ['h1_acc'] + model_names}
        for weight, weight_group in weight_groups:
            for model_name in ['h1_acc'] + model_names:
                col_name = model_name
                if model_name != 'h1_acc':
                    col_name += ' y'
                models_y[model_name].append(np.average(weight_group[col_name], weights=weight_group['len']))
        for i, j in models_y.items():
            models_y[i] = np.array(j)
        seed_h1_avg = models_y['h1_acc'][0]
        for model_name in model_names:
            model_y = models_y[model_name]

            # y_min = min(min(model_y), seed_h1_avg)
            # y_max = max(max(model_y), seed_h1_avg)
            # if y_max == y_min:
            #     # if seed_h1_avg >= seed_no_hist_0_avg_acc or y_max == y_min:
            #     continue

            # y_min = min(model_y)
            # y_max = max(model_y)

            # model_y_norm = (model_y - model_y.iloc[0]) * ((no_hist_0_avg_acc[model_name] - h1_avg_acc) /
            #                                               (y_max - y_min)) + no_hist_0_avg_acc[model_name]
            # model_y_norm = (model_y - model_y[0]) * ((no_hist_init_avg_acc - no_hist_final_avg_acc) /
            #                                          (y_max - y_min)) + no_hist_init_avg_acc

            # if model_name == 'no hist':
            #     y_max = model_y[0]
            #     y_min = model_y[-1]
            # model_y_norm = (model_y - y_max) * ((y_max_mean - y_min_mean) / (y_max - y_min)) + y_max_mean

            if model_name == 'no hist':
                no_hist_y = model_y
            # model_y_norm = (model_y - no_hist_y) * (y_max_mean - y_min_mean) / (no_hist_y[0] - no_hist_y[-1])
            model_y_norm = model_y - no_hist_y

            # if max(model_y) == min(model_y):
            #     print('max(y) = min(y)')

            # model_y_norm = (model_y - max(model_y)) * \
            #                ((model_y_init_avg_acc[model_name] - model_y_final_avg_acc[model_name]) /
            #                 (max(model_y) - min(model_y))) + model_y_init_avg_acc[model_name]

            df_dict['%s y' % model_name].extend(list(model_y_norm))

            # df_dict['%s y' % model_name].extend(list(model_y))

    df_norm = pd.DataFrame(df_dict)
    groups_by_weight = df_norm.groupby('weight')
    return [groups_by_weight.get_group(i) for i in weights]


def plot_results(log_dir, dataset, user_type, models, log_set, bin_size=1, user_name='', show_tradeoff_plots=False,
                 smooth_color_progression=False, std_opacity=0.15, performance_metric='accuracy', make_table=False):
    if user_name == '':
        df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s/logs/%s_%s.csv' % (log_dir, log_set, user_name))

    model_names = [i[:-2] for i in df_results.columns if ' x' in i and i[:-2] in models.keys()]
    xs, ys, xs_plot, ys_plot = [], [], [], []
    stds = {}
    autcs_average = []
    # autcs_by_seed = get_autcs_by_seed(df_results, model_names)
    h1_avg_acc = np.average(df_results['h1_acc'], weights=df_results['len'])
    weights = pd.unique(df_results['weight'])
    groups_by_weight = df_results.groupby('weight')

    if make_table:
        fig, (ax, tabax) = plt.subplots(nrows=2, figsize=(6.4, 4.8 + 0.3 * len(model_names)))
    cmap = plt.cm.get_cmap('jet')

    df_by_weight = [groups_by_weight.get_group(i) for i in weights]
    df_by_weight_norm = None
    for model_name in model_names:
        x = [np.average(i['%s x' % model_name], weights=i['len']) for i in df_by_weight]
        y = [np.average(i['%s y' % model_name], weights=i['len']) for i in df_by_weight]

        plot_std = models[model_name]['std']
        if plot_std:
            if df_by_weight_norm is None:
                no_hist_init_avg_acc = np.average(df_by_weight[0]['no hist y'], weights=df_by_weight[0]['len'])
                no_hist_final_avg_acc = np.average(df_by_weight[-1]['no hist y'], weights=df_by_weight[-1]['len'])
                model_names_for_std = [i for i in model_names if models[i]['std']]
                df_by_weight_norm = get_df_by_weight_norm(df_results, no_hist_init_avg_acc, no_hist_final_avg_acc,
                                                          weights, model_names_for_std)
            std = [df_by_weight_norm[i]['%s y' % model_name].std() for i in range(len(weights))]
        if compare_by_percentage:
            h1_area = (x[-1] - x[0]) * h1_avg_acc
            autc = auc(x, y) - h1_area
        else:
            autc = auc(x, y)
        autcs_average.append(autc)
        if model_name in ['no hist', 'model1']:
            no_hist_autc = autc

        xs.append(x)
        ys.append(y)
        if bin_size > 1:  # bin points
            last_idx = len(x) - ((len(x) - 2) % bin_size) - 1
            x_binned = np.mean(np.array(x[1:last_idx]).reshape(-1, bin_size), axis=1)
            y_binned = np.mean(np.array(y[1:last_idx]).reshape(-1, bin_size), axis=1)
            xs_plot.append([x[0]] + list(x_binned) + [np.mean(x[last_idx:-1]), x[-1]])
            ys_plot.append([y[0]] + list(y_binned) + [np.mean(y[last_idx:-1]), y[-1]])
            if plot_std:
                std_binned = np.mean(np.array(std[1:last_idx]).reshape(-1, bin_size), axis=1)
                stds[model_name] = [std[0]] + list(std_binned) + [np.mean(std[last_idx:-1]), std[-1]]
        else:
            xs_plot.append(x)
            ys_plot.append(y)
            if plot_std:
                stds[model_name] = std

    min_x = xs[0][0]
    max_x = xs[0][-1]
    h1_x = [min_x, max_x]
    h1_y = [h1_avg_acc, h1_avg_acc]
    if make_table:
        ax.plot(h1_x, h1_y, 'k--', marker='.', label='h1')
    else:
        plt.plot(h1_x, h1_y, 'k--', marker='.', label='pre-update model')

    autc_improvs = []
    for i in range(len(model_names)):
        # x = xs[i]
        # y = ys[i]
        # min_x_model = min(x)
        # if min_x_model > min_x:  # for models that start at better compatibility
        #     autc += (min_x_model - min_x) * (y[0] - h1_avg_acc)
        autc = autcs_average[i]
        if compare_by_percentage:
            autc_improvs.append((autc / no_hist_autc - 1) * 100)
        else:
            autc_improvs.append(autc - no_hist_autc)

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
            x_best, y_best, best_color, sign_best, autc_improv_best = x_plot, y_plot, color, sign, autc_improv
        else:
            # ax.plot(x_plot, y_plot, label=model_name, color=color, marker='.')
            if compare_by_percentage:
                label = '%s (%s%.1f%%)' % (model_name, sign, autc_improv)
            else:
                label = '%s (%.5f)' % (model_name, autc_improv)
            if make_table:
                ax.plot(x_plot, y_plot, label=label, color=color)
            else:
                if model_name == 'no hist':
                    plt.plot(x_plot, y_plot, label='baseline', color=color)
                else:
                    plt.plot(x_plot, y_plot, label=label, color=color)
            if model['std']:
                y = np.array(y_plot)
                s = np.array(stds[model_name])
                if make_table:
                    ax.fill_between(x_plot, y + s, y - s, facecolor=color, alpha=std_opacity)
                else:
                    plt.fill_between(x_plot, y + s, y - s, facecolor=color, alpha=std_opacity)
        if model_name == 'no hist':
            color = 'white'
        colors.append(color)
    if 'best_u' in model_names:
        if compare_by_percentage:
            label_best = 'best_u (%s%.1f%%)' % (sign_best, autc_improv_best)
        else:
            label_best = 'best_u (%.5f)' % autc_improv_best
        if make_table:
            ax.plot(x_best, y_best, label=label_best, color=best_color)
        else:
            plt.plot(x_best, y_best, label=label_best, color=best_color)
        if models['best_u']['std']:
            y = np.array(y_best)
            s = np.array(stds['best_u'])
            if make_table:
                ax.fill_between(x_best, y + s, y - s, facecolor=best_color, alpha=std_opacity)
            else:
                plt.fill_between(x_best, y + s, y - s, facecolor=best_color, alpha=std_opacity)

    if make_table:
        columns = ('gen', 'gen diss', 'hist', 'hist diss', '+ AUTC %')
        tabax.axis("off")
        tabax.table(cellText=cell_text, rowLabels=model_names_sorted, rowColours=colors, colLabels=columns,
                    loc='center')

    if user_name == '':
        # title = 'dataset=%s user_type=%s' % (dataset, user_type)
        if dataset == 'salaries':
            dataset = 'adult'
        elif dataset == 'recividism':
            dataset = 'recidivism'
        title = '%s dataset' % dataset
        save_name = '%s/%s_plots.png' % (log_dir, log_set)
    else:
        len_h = df_results.loc[0]['len']
        title = 'dataset=%s user=%s len(h)=%d' % (dataset, user_name, len_h)
        save_name = '%s/%s_%s.png' % (log_dir, log_set, user_name)
    if make_table:
        ax.set_xlabel('compatibility')
        ax.set_ylabel(performance_metric)
        ax.set_title(title)
    else:
        plt.xlabel('compatibility')
        plt.ylabel(performance_metric)
        plt.title(title)
        plt.legend()
    plt.savefig(save_name, bbox_inches='tight')
    if show_tradeoff_plots:
        plt.show()
    plt.clf()
    # return best_model


def get_best_models(log_dir, models, log_set, user_name='', plot_tradeoffs=False):
    if user_name == '':
        df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s/%s_%s.csv' % (log_dir, log_set, user_name))

    if plot_tradeoffs:
        seed_plots_dir = '%s/%s_seed_plots' % (log_dir, log_set)
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
        groups_by_weight = df_seed.groupby('weight')
        if user_name == '':
            h1_avg_acc = np.average(df_seed['h1_acc'], weights=df_seed['len'])
            dfs_by_weight = [groups_by_weight.get_group(i) for i in weights]
        else:
            h1_avg_acc = np.mean(df_seed['h1_acc'])
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
            color = models[model_name]['color']
            if model_name not in models.keys():
                continue
            autc = autcs[i]
            if best_autc is None or autc > best_autc:
                best_autc = autc
                best_model = model_name
            if plot_tradeoffs:
                plt.plot(xs_seed[i], ys_seed[i], label='%s autc=%.5f' % (model_name, autc), color=color)
        if plot_tradeoffs:
            plt.plot([min_x, max_x], [h1_avg_acc, h1_avg_acc], 'k--', label='h1', marker='.')
            plt.xlabel('compatibility')
            plt.ylabel('accuracy')
            plt.legend()
            plt.title('user=%s seed=%d best=%s' % (user_name, seed, best_model))
            plt.savefig('%s/user_%s seed_%d' % (seed_plots_dir, user_name, seed), bbox_inches='tight')
            plt.clf()
        best_models_by_seed.append(best_model)
    return seeds, best_models_by_seed
    # todo: return best model by weight


def summarize(log_dir, log_set, metrics, user_name=''):
    if user_name == '':
        df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s/%s_%s.csv' % (log_dir, log_set, user_name))
    model_names = [i[:-2] for i in df_results.columns if ' x' in i]

    # seeds = pd.unique(df_results['seed']).tolist()
    # groups_by_seed = df_results.groupby('seed')
    # weights = pd.unique(df_results['weight'])
    #
    # for seed_idx in range(len(seeds)):
    #     seed = seeds[seed_idx]
    #     print('\t%d/%d seed %d' % (seed_idx + 1, len(seeds), seed))
    #     df_seed = groups_by_seed.get_group(seed)
    #     groups_by_weight = df_seed.groupby('weight')
    #     if user_name == '':
    #         h1_avg_acc = np.average(df_seed['h1_acc'], weights=df_seed['len'])
    #         dfs_by_weight = [groups_by_weight.get_group(i) for i in weights]
    #     else:
    #         h1_avg_acc = np.mean(df_seed['h1_acc'])
    #         means = groups_by_weight.mean()
    #
    #     autcs = []
    #     for model_name in model_names:
    #         if user_name == '':
    #             x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
    #             y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]
    #         else:
    #             x = means['%s x' % model_name].tolist()
    #             y = means['%s y' % model_name].tolist()
    #
    #         h1_area = (x[-1] - x[0]) * h1_avg_acc
    #         autc = auc(x, y) - h1_area
    #         if model_name == 'no hist':
    #             no_hist_autc = autc
    #         autcs.append(autc)
    #
    #     for i in range(len(model_names)):
    #         autc_improv = autcs[i] / no_hist_autc - 1
    #         autc_improvs_by_seed[i].append(autc_improv)
    # return seeds, autc_improvs_by_seed, model_names

    weights = pd.unique(df_results['weight'])

    # avg
    groups_by_weight = df_results.groupby('weight')
    if user_name == '':
        h1_avg_acc = np.average(df_results['h1_acc'], weights=df_results['len'])
        dfs_by_weight = [groups_by_weight.get_group(i) for i in weights]
    else:
        h1_avg_acc = np.mean(df_results['h1_acc'])
        means = groups_by_weight.mean()
    autcs_avg = []
    for model_name in model_names:
        if user_name == '':
            x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
            y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]
        else:
            x = means['%s x' % model_name].tolist()
            y = means['%s y' % model_name].tolist()

        if compare_by_percentage:
            h1_area = (x[-1] - x[0]) * h1_avg_acc
            autc_avg = auc(x, y) - h1_area
        else:
            autc_avg = auc(x, y)
        if model_name == 'no hist':
            no_hist_autc_avg = autc_avg
        autcs_avg.append(autc_avg)

    # std
    no_hist_autc_seeds = []
    autcs_seed = []
    for seed, df_seed in df_results.groupby('seed'):
        autcs_seed.append([])
        groups_by_weight = df_seed.groupby('weight')
        if user_name == '':
            h1_avg_acc = np.average(df_seed['h1_acc'], weights=df_seed['len'])
            dfs_by_weight = [groups_by_weight.get_group(i) for i in weights]
        else:
            h1_avg_acc = np.mean(df_seed['h1_acc'])
            means = groups_by_weight.mean()
        for model_name in model_names:
            if user_name == '':
                x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
                y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]
            else:
                x = means['%s x' % model_name].tolist()
                y = means['%s y' % model_name].tolist()

            if compare_by_percentage:
                h1_area = (x[-1] - x[0]) * h1_avg_acc
                autc_seed = auc(x, y) - h1_area
            else:
                autc_seed = auc(x, y)
            if model_name == 'no hist':
                no_hist_autc_seeds.append(autc_seed)
            autcs_seed[-1].append(autc_seed)
    autcs_seed, no_hist_autc_seeds = np.array(autcs_seed), np.array(no_hist_autc_seeds)

    results = []
    if 'avg' in metrics:
        autc_improvs_avg = [[] for i in range(len(model_names))]
        for i in range(len(model_names)):
            if compare_by_percentage:
                autc_improv_avg = autcs_avg[i] / no_hist_autc_avg - 1
            else:
                autc_improv_avg = autcs_avg[i] - no_hist_autc_avg
            autc_improvs_avg[i].append(autc_improv_avg)
        results.append(autc_improvs_avg)
    if 'std' in metrics:
        if compare_by_percentage:
            autc_improvs_std = np.std(autcs_seed.T / no_hist_autc_seeds - 1, axis=1)
        else:
            autc_improvs_std = np.std(autcs_seed.T - no_hist_autc_seeds, axis=1)
        autc_improvs_std = autc_improvs_std.reshape(len(autc_improvs_std), 1).tolist()
        results.append(autc_improvs_std)
    if 'paired_ttest' in metrics:
        autc_pval = [[] for i in range(len(model_names))]
        for i, model_autc_seeds in enumerate(autcs_seed.T):
            t_stat, p_val = ttest_rel(model_autc_seeds, no_hist_autc_seeds)
            autc_pval[i].append(p_val)
        results.append(autc_pval)

    return results + [model_names]


def add_best_model(log_dir, valid_set, test_set):
    df_best = pd.read_csv('%s/best_models_%s.csv' % (log_dir, valid_set))
    df_test = pd.read_csv('%s/%s_log.csv' % (log_dir, test_set))
    groups_by_user = df_test.groupby('user')
    user_names = pd.unique(df_best['user'])
    seeds = pd.unique(df_best['seed'])
    # best_models = df_best.to_numpy()
    best_models = {user: [list(row) for i, row in data.iterrows()] for user, data in df_best.groupby('user')}
    new_model_x = []
    new_model_y = []
    # i = 0
    for user_idx, user_name in enumerate(user_names):
        print('\tuser %d/%d' % (user_idx + 1, len(user_names)))
        df_user = groups_by_user.get_group(user_name)
        groups_by_seed = df_user.groupby('seed')
        for seed_idx, seed in enumerate(seeds):
            df_seed = groups_by_seed.get_group(seed)
            best_user, best_seed, best_model = best_models[user_name][seed_idx]
            new_model_x.extend(df_seed['%s x' % best_model].tolist())
            new_model_y.extend(df_seed['%s y' % best_model].tolist())
            # print('(%s=%s) (%d=%d)' % (user_name, best_user, seed, best_seed))
            if user_name != best_user or seed != best_seed:
                raise ValueError('results and best lists not in same order of user -> seed')
            # i += 1

    df_test['best_u x'] = new_model_x
    df_test['best_u y'] = new_model_y
    df_test.to_csv('%s/%s_with_best_log.csv' % (log_dir, test_set), index=False)


def binarize_results_by_compat_values(log_dir, log_set, num_bins=100):
    bins = np.array([i / num_bins for i in range(num_bins + 1)])
    df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    dict_binarized = {i: [] for i in df_results.columns}
    user_names = pd.unique(df_results['user'])
    groups_by_user = df_results.groupby('user')
    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    seeds = None
    inner_seeds = None
    missing_values = []

    for user_idx, user_name in enumerate(user_names):
        print('%d/%d user=%s' % (user_idx + 1, len(user_names), user_name))
        df_user = groups_by_user.get_group(user_name)
        user_len = df_user['len'].iloc[0]
        user_name_repeated = [user_name] * (num_bins + 1)
        user_len_repeated = [user_len] * (num_bins + 1)
        if seeds is None:
            seeds = pd.unique(df_user['seed'])
        groups_by_seed = df_user.groupby('seed')

        for seed_idx, seed in enumerate(seeds):
            print('\t%d/%d seed=%d' % (seed_idx + 1, len(seeds), seed))
            seed_repeated = [seed] * (num_bins + 1)
            try:
                df_seed = groups_by_seed.get_group(seed)
            except KeyError:
                missing_values.append('user=%s seed=%d' % (user_name, seed))
                continue
            if inner_seeds is None:
                inner_seeds = pd.unique(df_seed['inner_seed'])
            groups_by_inner_seed = df_seed.groupby('inner_seed')

            for inner_seed_idx, inner_seed in enumerate(inner_seeds):
                try:
                    df_inner_seed = groups_by_inner_seed.get_group(inner_seed)
                except KeyError:
                    missing_values.append('user=%s seed=%d inner_seed=%d' % (user_name, seed, inner_seed))
                    continue
                if len(missing_values) != 0:
                    continue
                h1_acc = df_inner_seed['h1_acc'].iloc[0]
                inner_seed_repeated = [inner_seed] * (num_bins + 1)
                h1_acc_repeated = [h1_acc] * (num_bins + 1)

                dict_binarized['user'].extend(user_name_repeated)
                dict_binarized['len'].extend(user_len_repeated)
                dict_binarized['seed'].extend(seed_repeated)
                dict_binarized['inner_seed'].extend(inner_seed_repeated)
                dict_binarized['h1_acc'].extend(h1_acc_repeated)
                dict_binarized['weight'].extend(bins)
                xs = []
                ys = []
                for model_name in model_names:
                    x = df_inner_seed['%s x' % model_name].tolist()
                    y = df_inner_seed['%s y' % model_name].tolist()
                    for i in range(1, len(x)):  # make x monotonically increasing
                        if x[i] < x[i - 1]:
                            x[i] = x[i - 1]
                    xs.append(x)
                    ys.append(y)

                # add min and max x to each model
                min_x = min([min(i) for i in xs])
                max_x = max([max(i) for i in xs])
                x_bins = (bins * (max_x - min_x) + min_x).tolist()
                # to avoid floating point weirdness in first and last values
                x_bins[0] = min_x
                x_bins[-1] = max_x
                for i in range(len(model_names)):
                    x = xs[i]
                    y = ys[i]
                    xs[i] = [min_x] + x + [x[-1], max_x]
                    ys[i] = [y[0]] + y + [h1_acc, h1_acc]

                # binarize
                for model_idx in range(len(model_names)):
                    model_name = model_names[model_idx]
                    x = xs[model_idx]
                    y = ys[model_idx]
                    y_bins = []
                    j = 0
                    for x_bin in x_bins:  # get y given x for each x_bin
                        while not x[j] <= x_bin <= x[j + 1]:
                            j += 1
                        x_left, x_right, y_left, y_right = x[j], x[j + 1], y[j], y[j + 1]
                        if x_left == x_right:  # vertical line
                            y_bin = max(y_left, y_right)
                        else:
                            slope = (y_right - y_left) / (x_right - x_left)
                            y_bin = y_left + slope * (x_bin - x_left)
                        y_bins.append(y_bin)
                    dict_binarized['%s x' % model_name].extend(x_bins)
                    dict_binarized['%s y' % model_name].extend(y_bins)
    if len(missing_values) != 0:
        with open('%s/%s_missing_values.txt' % (log_dir, log_set), 'w') as file:
            for missing_value in missing_values:
                file.write('%s\n' % missing_value)
        print(missing_values)
        raise KeyError('missing values!')
    pd.DataFrame(dict_binarized).to_csv('%s/%s_bins_log.csv' % (log_dir, log_set), index=False)


def best_count_values(log_dir, log_set):
    df_best = pd.read_csv('%s/best_models_%s.csv' % (log_dir, log_set))
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
    df_result.to_csv('%s/best_models_%s_counts.csv' % (log_dir, log_set), index=False)


def get_autcs_averaged_over_inner_seeds(log_dir, log_set):
    df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    users = pd.unique(df_results['user'])
    user_groups = df_results.groupby('user')
    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    with open('%s/%s_autcs.csv' % (log_dir, log_set), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user', 'len', 'seed'] + [i for i in model_names])
        for user_idx, user in enumerate(users):
            print('\tuser %d/%d' % (user_idx + 1, len(users)))
            df_user = user_groups.get_group(user)
            hist_len = df_user.iloc[0]['len']
            for seed, df_seed in df_user.groupby('seed'):
                row = [user, hist_len, seed]
                means = df_seed.groupby('weight').mean()
                for model_name in model_names:
                    x = means['%s x' % model_name].tolist()
                    y = means['%s y' % model_name].tolist()
                    row += [auc(x, y)]
                writer.writerow(row)


def get_all_autcs(log_dir, log_set):
    df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    users = pd.unique(df_results['user'])
    user_groups = df_results.groupby('user')
    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    with open('%s/%s_all_autcs.csv' % (log_dir, log_set), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user', 'seed', 'inner seed'] + [i for i in model_names])
        for user_idx, user in enumerate(users):
            print('\tuser %d/%d' % (user_idx + 1, len(users)))
            df_user = user_groups.get_group(user)
            for seed, df_seed in df_user.groupby('seed'):
                for inner_seed, df_inner_seed in df_seed.groupby('inner_seed'):
                    row = [user, seed, inner_seed]
                    for model_name in model_names:
                        x = df_inner_seed['%s x' % model_name].tolist()
                        y = df_inner_seed['%s y' % model_name].tolist()
                        row += [auc(x, y)]
                    writer.writerow(row)


def get_user_distances(log_dir):
    wasserstein_distances = pd.read_csv('%s/wasserstein_distances.csv' % log_dir, index_col='user')
    feature_importances = pd.read_csv('%s/feature_importances.csv' % log_dir, index_col='user')
    gen_feature_importance = feature_importances.loc['general']
    user_distances = {}
    for user, row in wasserstein_distances.iterrows():
        user_distances[user] = np.average(row, weights=gen_feature_importance)
    return user_distances


def get_best_from_row(rows, models, append_vector=True):
    bests = []
    for row in rows:
        best_autc = 0
        best_name = 'no hist'
        best_idx = 0
        for i, model in enumerate(models):
            if row[model] > best_autc:
                best_autc = row[model]
                best_name = model
                best_idx = i
        if append_vector:
            best_vector = [0] * len(models)
            best_vector[best_idx] = 1
            bests.append([best_name, best_vector])
        else:
            bests.append(best_name)
    return bests


def get_sample_indexes_from_user_indexes(fold, num_seeds):
    train_index, test_index = [], []
    for users_index, subset_index in zip(fold, [train_index, test_index]):
        for i in users_index:
            start_index = i * num_seeds
            subset_index.extend([start_index + j for j in range(num_seeds)])
    return np.array(train_index), np.array(test_index)


def get_NN_meta_model(X, y, dense_lens):
    input = Input(shape=(X.shape[1],), name='input')
    layer = input
    for dense_len in dense_lens:
        layer = Dense(dense_len, activation='relu')(layer)
    output = Dense(y.shape[1], activation='softmax')(layer)
    model = Model(input, output, name='meta-model')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # model.summary()
    return model


def execute_phase(phase, log_set):
    binarize_by_compat = False
    individual_users = False
    get_best = False
    get_autoML_best = False
    add_best = False
    make_summary = False
    get_autcs = False
    count_best = False
    test_set = 'test_bins'

    print('\n%s' % phase)
    if phase == 'binarize validation results':
        log_set = 'valid'
        binarize_by_compat = True
    elif phase == 'binarize train results':
        log_set = 'train'
        binarize_by_compat = True
    elif phase == 'get best_u for each user using binarized validation results':
        log_set = 'valid_bins'
        individual_users = True
        get_best = True
    elif phase == 'get best_u for each user using binarized train results':
        log_set = 'train_bins'
        individual_users = True
        get_best = True
    elif phase == 'binarize test results':
        log_set = 'test'
        binarize_by_compat = True
    elif phase == 'add best_u computed from validation to binarized test results':
        log_set = 'valid_bins'
        add_best = True
    elif phase == 'add best_u computed from train to binarized validation results':
        log_set = 'train_bins'
        test_set = 'valid_bins'
        add_best = True
    elif phase == 'generate averaged plots for binarized test results with best':
        log_set = 'test_bins_with_best'
    elif phase == 'generate individual user plots for test bins with best results':
        log_set = 'test_bins_with_best'
        individual_users = True
    elif phase == 'create test summary':
        log_set = 'test_bins_with_best'
        make_summary = True
        individual_users = True
    elif phase == 'generate user plots for binarized validation results':
        log_set = 'valid_bins'
        individual_users = True
    elif phase == 'generate averaged plots for binarized validation results':
        log_set = 'valid_bins'
    elif phase == 'generate averaged plots for binarized test results':
        log_set = 'test_bins'
    elif phase == 'binarize train results':
        log_set = 'train'
        binarize_by_compat = True
    elif phase == 'generate averaged plots for binarized train results':
        log_set = 'train_bins'
    elif phase == 'get autcs averaged over inner seeds for train bins':
        log_set = 'train_bins'
        get_autcs = True
    elif phase == 'get autcs averaged over inner seeds for validation bins':
        log_set = 'valid_bins'
        get_autcs = True
    elif phase == 'get autcs averaged over inner seeds for test bins':
        log_set = 'test_bins'
        get_autcs = True
    elif phase == 'get autcs averaged over inner seeds for test bins with best':
        log_set = 'test_bins_with_best'
        get_autcs = True
    elif phase == 'generate averaged plots for binarized validation results with best':
        log_set = 'valid_bins_with_best'
    elif phase == 'get best for each user':
        individual_users = True
        get_autoML_best = True

    results_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/results/%s' % model_type
    log_dir = '%s/%s/%s/%s/%s' % (results_dir, dataset, version, user_type, performance_metric)
    models = get_model_dict('jet')

    if add_best:
        add_best_model(log_dir, log_set, test_set)
    elif count_best:
        best_count_values(log_dir, log_set)
    elif binarize_by_compat:
        binarize_results_by_compat_values(log_dir, log_set, num_normalization_bins)
    elif get_autcs:
        get_autcs_averaged_over_inner_seeds(log_dir, log_set)
        # get_all_autcs(log_dir, log_set)
    elif not individual_users:  # make sure this is last elif
        if get_best:
            print('got best models for general set, not individual users!')
        else:
            plot_results(log_dir, dataset, user_type, models, log_set, bin_size=bin_size, show_tradeoff_plots=True)
    else:
        if get_autoML_best:
            models = ['no hist', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
            make_data_analysis_per_inner_seed(log_dir, dataset, user_type, target_col)
            train_autcs = pd.read_csv('%s/train_bins_autcs.csv' % log_dir)
            valid_autcs = pd.read_csv('%s/valid_bins_autcs.csv' % log_dir)
            with open('%s/best_models_valid_bins.csv' % log_dir, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['user', 'seed', 'model'])
                if meta_learning_version == 'generalize train to validation':
                    for row_idx in range(len(train_autcs)):
                        train_row, valid_row = train_autcs.iloc[row_idx], valid_autcs.iloc[row_idx]
                        train_best_autc = 0
                        train_best_model = 'no hist'
                        for model in models:
                            if train_row[model] > train_best_autc:
                                train_best_autc = train_row[model]
                                train_best_model = model
                        if valid_row[train_best_model] > valid_row['no hist']:
                            writer.writerow([train_row['user'], train_row['seed'], train_best_model])
                        else:
                            writer.writerow([train_row['user'], train_row['seed'], 'no hist'])
                elif meta_learning_version == 'meta-learner split within users':
                    test_autcs = pd.read_csv('%s/test_bins_autcs.csv' % log_dir)
                    distances = pd.read_csv('%s/distances_by_seed.csv' % log_dir)
                    seeds = pd.unique(train_autcs['seed'])
                    train_groups_by_seed = train_autcs.groupby('seed')
                    valid_groups_by_seed = valid_autcs.groupby('seed')
                    test_groups_by_seed = test_autcs.groupby('seed')
                    distances_groups_by_seed = distances.groupby('seed')
                    max_depth = 5
                    for seed_idx, seed in enumerate(seeds):
                        train_autcs_seed = train_groups_by_seed.get_group(seed)
                        valid_autcs_seed = valid_groups_by_seed.get_group(seed)
                        test_autcs_seed = test_groups_by_seed.get_group(seed)
                        distances_seed = distances_groups_by_seed.get_group(seed)
                        x_train, y_train, x_test, y_test = [], [], [], []
                        for row_idx in range(len(train_autcs_seed)):
                            train_row = train_autcs_seed.iloc[row_idx]
                            valid_row = valid_autcs_seed.iloc[row_idx]
                            test_row = test_autcs_seed.iloc[row_idx]
                            d = distances_seed.iloc[row_idx]
                            l = train_row['len']
                            train_best, valid_best, test_best = get_best_from_row(
                                [train_row, valid_row, test_row], models)
                            # todo: maybe append train_best too
                            x_train_row = [l, d['train_to_h2'], d['train_to_valid'], d['h2_to_valid']] + train_best[1]
                            x_test_row = [l, d['train_to_h2'], d['train_to_test'], d['h2_to_test']] + valid_best[1]
                            x_train.append(x_train_row)
                            x_test.append(x_test_row)
                            y_train.append(valid_best[0])
                            y_test.append(test_best[0])
                        # train meta-model
                        # max_depth += 1
                        x_train = np.array(x_train)
                        x_test = np.array(x_test)
                        y_train = np.array(y_train)
                        y_test = np.array(y_test)

                        meta_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, random_state=1),
                                                        n_estimators=1000, learning_rate=1, random_state=1)
                        meta_model.fit(x_train, y_train)

                        # meta_model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
                        # meta_model.fit(x_train, y_train)
                        # # feature_names = ['len', 'train_to_h2', 'train_to_test', 'h2_to_test'] + models
                        # # tree.plot_tree(meta_model, feature_names=feature_names)
                        # # plt.show()

                        y_predicted = meta_model.predict(x_test)
                        train_acc = meta_model.score(x_train, y_train)
                        test_acc = meta_model.score(x_test, y_test)

                        for row_idx in range(len(train_autcs_seed)):
                            row = train_autcs_seed.iloc[row_idx]
                            predicted_label = y_predicted[row_idx]
                            writer.writerow([row['user'], seed, predicted_label])

                        print('seed %d/%d max_depth=%d train_score=%.4f test_score=%.4f'
                              % (seed_idx + 1, len(seeds), max_depth, train_acc, test_acc))
                elif meta_learning_version == 'meta-learner split between users':
                    meta_dataset = train_autcs[['user', 'len']].copy()

                    # add distances
                    distances = pd.read_csv('%s/distances_by_seed.csv' % log_dir)
                    distance_cols = list(distances.columns)[3:]
                    for distance_col in distance_cols:
                        meta_dataset[distance_col] = distances[distance_col]

                    # add best models
                    test_autcs = pd.read_csv('%s/test_bins_autcs.csv' % log_dir)

                    if meta_learning_task == 'selecting best':
                        bests_train, bests_valid, bests_test = [], [], []
                        bests_train_vectors, bests_valid_vectors, bests_test_vectors = [], [], []
                        for row_idx in range(len(train_autcs)):
                            train_row, valid_row, test_row = [
                                train_autcs.iloc[row_idx], valid_autcs.iloc[row_idx], test_autcs.iloc[row_idx]]
                            train_best, valid_best, test_best = get_best_from_row(
                                [train_row, valid_row, test_row], models)
                            # append model names
                            bests_train.append(train_best[0])
                            bests_valid.append(valid_best[0])
                            bests_test.append(test_best[0])
                            # append one hot vectors
                            bests_train_vectors.append(train_best[1])
                            bests_valid_vectors.append(valid_best[1])
                            bests_test_vectors.append(test_best[1])
                        meta_dataset['best_train'] = bests_train
                        meta_dataset['best_valid'] = bests_valid
                        meta_dataset['best_test'] = bests_test
                        meta_dataset.to_csv('%s/meta_dataset_ver_1.csv' % log_dir, index=False)
                        X = meta_dataset.drop(columns=['user', 'best_train', 'best_valid', 'best_test']).to_numpy()
                        X = np.concatenate([X, bests_train_vectors, bests_valid_vectors], axis=1)  # encode model names
                        if meta_learning_model == 'NN':
                            y = np.array(bests_test_vectors)
                        else:
                            y = np.array(bests_test)

                    elif meta_learning_task == 'selecting best from train or valid':
                        # score(baseline, train), score(baseline, valid),
                        # score(best_train, valid) and score(best_valid, train)
                        scores = [[], [], [], []]
                        # string and vector
                        labels = [[], []]
                        for row_idx in range(len(train_autcs)):
                            train_row, valid_row, test_row = [
                                train_autcs.iloc[row_idx], valid_autcs.iloc[row_idx], test_autcs.iloc[row_idx]]
                            train_best, valid_best, test_best = get_best_from_row(
                                [train_row, valid_row, test_row], models, append_vector=False)
                            scores[0].append(train_row['no hist'] / train_row[train_best])
                            scores[1].append(valid_row['no hist'] / valid_row[valid_best])
                            scores[2].append(valid_row[train_best] / valid_row[valid_best])
                            scores[3].append(train_row[valid_best] / train_row[train_best])

                            # if valid_best == test_best:
                            #     labels[0].append('valid')
                            #     labels[1].append([0, 0, 1])
                            # elif train_best == test_best:
                            #     labels[0].append('train')
                            #     labels[1].append([0, 1, 0])
                            # else:
                            #     labels[0].append('baseline')
                            #     labels[1].append([1, 0, 0])

                            if test_row[valid_best] >= test_row[train_best]:
                                if test_row[valid_best] >= test_row['no hist']:  # select valid_best
                                    labels[0].append('valid')
                                    labels[1].append([0, 0, 1])
                                else:  # select baseline
                                    labels[0].append('baseline')
                                    labels[1].append([1, 0, 0])
                            else:  #



                        meta_dataset['score(baseline, train)'] = scores[0]
                        meta_dataset['score(baseline, valid)'] = scores[1]
                        meta_dataset['score(best_train, valid)'] = scores[2]
                        meta_dataset['score(best_valid, train)'] = scores[3]
                        meta_dataset['label'] = labels[0]
                        meta_dataset.to_csv('%s/meta_dataset_ver_3.csv' % log_dir, index=False)
                        X = meta_dataset.drop(columns=['user', 'label']).to_numpy()
                        if meta_learning_model == 'NN':
                            y = np.array(labels[1])
                            X[:, 0] = X[:, 0] / np.max(X[:, 0])
                        else:
                            y = np.array(labels[0])

                    # cross validation over users
                    print('starting cross-validation...')
                    meta_learning_results = []
                    users = pd.unique(meta_dataset['user'])
                    num_seeds = int(len(meta_dataset) / len(users))
                    k_fold_cross_validation = KFold(n_splits=meta_cross_validation_splits,
                                                    shuffle=True, random_state=1)

                    ccp_alphas = [i / 10000 for i in range(1, 10)]
                    ccp_alphas += [i / 1000 for i in range(1, 10)]
                    ccp_alphas += [i / 100 for i in range(1, 10)]
                    ccp_alphas += [i / 10 for i in range(1, 10)]

                    for fold_idx, fold in enumerate(k_fold_cross_validation.split(users)):
                        train_index, test_index = get_sample_indexes_from_user_indexes(fold, num_seeds)
                        X_train, y_train = X[train_index], y[train_index]
                        X_test, y_test = X[test_index], y[test_index]

                        if meta_learning_model == 'AdaBoost':
                            # meta_model = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=10, random_state=1),
                            #                                 n_estimators=200, learning_rate=1, random_state=1)
                            # meta_model.fit(X_train, y_train)

                            label_names, label_counts = np.unique(y_test, return_counts=True)
                            test_fractions = pd.DataFrame(label_counts.reshape(1, 3) / np.sum(label_counts),
                                                          columns=label_names)

                            print('fold=%d' % (fold_idx + 1))
                            for ccp_alpha in ccp_alphas:
                                meta_model = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=1)
                                meta_model.fit(X_train, y_train)

                                # # plot
                                # feature_names = meta_dataset.columns.drop(['user', 'label'])
                                # fig, ax = plt.subplots(figsize=(30, 25))
                                # tree.plot_tree(meta_model, feature_names=feature_names, fontsize=6)
                                # plt.savefig('meta_tree', dpi=200)
                                # # plt.show()

                                train_acc = meta_model.score(X_train, y_train)
                                test_acc = meta_model.score(X_test, y_test)

                                meta_learning_results.append(
                                    [fold_idx, test_fractions['baseline'][0], test_fractions['train'][0],
                                     test_fractions['valid'][0], ccp_alpha, train_acc, test_acc])
                            # break

                        elif meta_learning_model == 'NN':
                            meta_model = get_NN_meta_model(X, y, [5])
                            history = meta_model.fit(X_train, y_train, batch_size=100, epochs=200, verbose=2)
                            train_loss, train_acc = meta_model.evaluate(X_train, y_train, verbose=0)
                            test_loss, test_acc = meta_model.evaluate(X_test, y_test, verbose=0)

                            y_proba = meta_model.predict(X_test)
                            matrix = confusion_matrix(y_test.argmax(1), y_proba.argmax(1))
                            print(matrix)

                        # print('fold %d train_acc=%.4f test_acc=%.4f | baselines=%.4f best_trains=%.4f best_valids=%.4f'
                        #       % (fold_idx + 1, train_acc, test_acc, test_fractions['baseline'],
                        #          test_fractions['train'], test_fractions['valid']))

                        # break
                    pd.DataFrame(meta_learning_results,
                                 columns=['fold', 'baselines_fraction', 'best_trains_fraction', 'best_valids_fraction',
                                          'ccp_alpha', 'train_acc', 'test_acc']).to_csv(
                        '%s/meta_learning_results.csv' % log_dir, index=False)
        else:
            users_dir = '%s/users_%s' % (log_dir, log_set)
            user_logs_dir = '%s/logs' % users_dir
            if not os.path.exists(users_dir):
                safe_make_dir(user_logs_dir)
                split_users(log_dir, log_set)
            df = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
            user_ids = pd.unique(df['user'])
            user_groups = df.groupby('user')
            lens = [user_groups.get_group(i)['len'].iloc[0] for i in user_ids]
            if get_best:
                user_col = []
                seed_col = []
                model_col = []
                for user_idx in range(len(user_ids)):
                    user_id = user_ids[user_idx]
                    print('%d/%d user %s' % (user_idx + 1, len(user_ids), user_id))
                    seeds, best_models_by_seed = get_best_models('%s/users_%s/logs' % (log_dir, log_set), models,
                                                                 log_set,
                                                                 user_name=user_id)
                    user_col += [user_id] * len(seeds)
                    seed_col += seeds
                    model_col += best_models_by_seed
                df = pd.DataFrame({'user': user_col, 'seed': seed_col, 'model': model_col})
                df.to_csv('%s/best_models_%s.csv' % (log_dir, log_set), index=False)
            elif make_summary:
                if compare_by_percentage:
                    method = 'percent'
                else:
                    method = 'absolute'
                make_data_analysis(log_dir, dataset, user_type, target_col)
                user_col = []
                len_col = []
                distance_col = []
                user_distances = get_user_distances(log_dir)
                summary_final_results = [None for i in summary_metrics]
                for user_idx in range(len(user_ids)):
                    user_id = user_ids[user_idx]
                    l = lens[user_idx]
                    print('%d/%d user %s' % (user_idx + 1, len(user_ids), user_id))
                    summary_results = summarize('%s/users_%s/logs' % (log_dir, log_set), log_set, summary_metrics,
                                                user_name=user_id)
                    user_col += [user_id]  # * len(seeds)
                    len_col += [l]  # * len(seeds)
                    distance_col += [user_distances[user_id]]
                    if summary_final_results[0] is None:
                        for row_idx in range(len(summary_metrics)):
                            summary_final_results[row_idx] = summary_results[row_idx]
                    else:
                        for row_idx in range(len(summary_metrics)):
                            for j in range(len(summary_results[0])):
                                summary_final_results[row_idx][j].extend(summary_results[row_idx][j])
                # all users together
                summary_results = summarize(log_dir, log_set, summary_metrics)
                model_names = summary_results[-1]
                user_col += ['all users']
                len_col += [sum(len_col)]
                distance_col += [0]
                for row_idx in range(len(summary_metrics)):
                    for j in range(len(summary_results[0])):
                        summary_final_results[row_idx][j].extend(summary_results[row_idx][j])

                for metric, metric_results in zip(summary_metrics, summary_final_results):
                    df_dict = {'user': user_col, 'len': len_col, 'dist': distance_col}
                    for row_idx in range(len(model_names)):
                        df_dict[model_names[row_idx]] = metric_results[row_idx]
                    df = pd.DataFrame(df_dict)
                    df.to_csv('%s/summary_of_%s_%s_%s.csv' % (log_dir, log_set, method, metric), index=False)
            else:
                for user_idx in range(len(user_ids)):
                    user_id = user_ids[user_idx]
                    print('%d/%d user=%s' % (user_idx + 1, len(user_ids), user_id))
                    plot_results('%s/users_%s' % (log_dir, log_set), dataset, user_type, models, log_set,
                                 bin_size=bin_size, user_name=user_id)


# todo: CHOOSE WHAT TO ANALYSE
# dataset = 'ednet'
dataset = 'assistment'
# dataset = 'salaries'
# dataset = 'recividism'
# dataset = 'averaging tradeoffs'

# use_autoML = False
use_autoML = True

if use_autoML:
    phases = [
        # 'binarize train results',
        # 'binarize validation results',
        # 'binarize test results',
        # 'get autcs averaged over inner seeds for train bins',
        # 'get autcs averaged over inner seeds for validation bins',
        # 'get autcs averaged over inner seeds for test bins',
        'get best for each user',
        # 'add best_u computed from validation to binarized test results',
        # 'generate averaged plots for binarized test results with best',
        # 'create test summary',

        # 'generate averaged plots for binarized train results',
        # 'generate averaged plots for binarized validation results with best',
        # 'get autcs averaged over inner seeds for test bins with best',
    ]
else:
    phases = [
        # 'binarize validation results',  # phase 1
        # 'get best_u for each user using binarized validation results',  # phase 2
        # 'binarize test results',  # phase 3
        # 'add best_u computed from validation to binarized test results',  # phase 4
        # 'generate averaged plots for binarized test results with best',  # phase 5
        # 'generate individual user plots for test bins with best results',  # phase 6
        # 'create test summary',  # phase 7
        # 'generate user plots for binarized validation results',
        # 'generate averaged plots for binarized validation results',
        # 'generate averaged plots for binarized test results',
        # 'binarize train results',
        # 'generate averaged plots for binarized train results',
        # 'get autcs averaged over inner seeds',
    ]

add_parametrized_models = True
num_normalization_bins = 20
compare_by_percentage = True
# meta_learning_version = 'best from validation'
# meta_learning_version = 'generalize train to validation'
# meta_learning_version = 'meta-learner split within users'
meta_learning_version = 'meta-learner split between users'
# meta_learning_task = 'selecting best'
meta_learning_task = 'selecting best from train or valid'
# meta_learning_model = 'NN'
meta_learning_model = 'AdaBoost'
meta_cross_validation_splits = 10
# summary_metrics = ['avg', 'std', 'paired_ttest']
summary_metrics = ['avg']
log_set = 'test_bins_with_best'

if meta_learning_model == 'NN':
    from keras.models import Model
    from keras.layers import Dense, Input

version, user_type, target_col, model_type, performance_metric, bin_size = get_experiment_parameters(dataset, True)
print('dataset = %s' % dataset)
for phase in phases:
    execute_phase(phase, log_set)

print('\ndone')
