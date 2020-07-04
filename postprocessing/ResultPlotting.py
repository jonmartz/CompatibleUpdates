import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from postprocessing.DataAnalysis import make_data_analysis


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_users(log_dir, log_set):
    df_log = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    log_by_users = df_log.groupby('user')

    for user_id, user_data in log_by_users:
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
        'no hist': {'sample_weight': [1, 1, 0, 0], 'color': 'black', 'std': False},
        # 'sim_ann': {'sample_weight': [0.0, 0.6352, 0.3119, 0.0780], 'color': 'purple'},
        # 'hybrid': {'sample_weight': ['', '', '', ''], 'color': 'green', 'std': False},
        'best_u': {'sample_weight': ['', '', '', ''], 'color': 'red', 'std': False},
        # SYNTHETIC:
        # 'model1': {'sample_weight': ['', '', '', ''], 'color': 'red', 'std': True},
        # 'model2': {'sample_weight': ['', '', '', ''], 'color': 'blue', 'std': True},
    }
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


def get_autcs_by_seed(df_results, model_names):
    autcs_by_seed = []
    groups_by_seed = df_results.groupby('seed')
    for seed, df_seed in groups_by_seed:
        groups_by_weight = df_seed.groupby('weight')
        # todo: here


def plot_results(log_dir, dataset, user_type, models, log_set, bin_size=1, user_name='', show_tradeoff_plots=False,
                 smooth_color_progression=False, std_opacity=0.15, performance_metric='accuracy', make_table=False):
    if user_name == '':
        df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s/logs/%s_%s.csv' % (log_dir, log_set, user_name))

    # df_results = df_results[~df_results['user'].isin(skip_users)]

    model_names = [i[:-2] for i in df_results.columns if ' x' in i and i[:-2] in models.keys()]
    xs, ys, xs_plot, ys_plot = [], [], [], []
    stds = {}
    autcs_average = []
    autcs_by_seed = get_autcs_by_seed(df_results, model_names)
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

                # model_y_init_avg_acc = {
                #     i: np.average(df_by_weight[0]['%s y' % i], weights=df_by_weight[0]['len'])
                #     for i in model_names
                # }
                # model_y_final_avg_acc = {
                #     i: np.average(df_by_weight[-1]['%s y' % i], weights=df_by_weight[-1]['len'])
                #     for i in model_names
                # }
                # model_names_for_std = [i for i in model_names if models[i]['std']]
                # df_by_weight_norm = get_df_by_weight_norm(df_results.drop(columns=['%s x' % i for i in model_names_for_std]),
                #                                           model_y_init_avg_acc, model_y_final_avg_acc,
                #                                           weights, model_names)

            # # todo: std by user
            # var = [np.average((df_by_weight_norm[i]['%s y' % model_name] - y[i]) ** 2,
            #                   weights=df_by_weight_norm[i]['len']) for i in range(len(weights))]
            # std = [math.sqrt(i) for i in var]

            # todo: std by seed
            std = [df_by_weight_norm[i]['%s y' % model_name].std() for i in range(len(weights))]

        h1_area = (x[-1] - x[0]) * h1_avg_acc
        autc = auc(x, y) - h1_area

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
    # min_x = min(min(i) for i in xs)
    # max_x = max(max(i) for i in xs)
    h1_x = [min_x, max_x]
    h1_y = [h1_avg_acc, h1_avg_acc]
    if make_table:
        ax.plot(h1_x, h1_y, 'k--', marker='.', label='h1')
    else:
        plt.plot(h1_x, h1_y, 'k--', marker='.', label='pre-update model')

    autc_improvs = []
    for i in range(len(model_names)):
        x = xs[i]
        y = ys[i]
        autc = autcs_average[i]
        min_x_model = min(x)
        if min_x_model > min_x:  # for models that start at better compatibility
            autc += (min_x_model - min_x) * (y[0] - h1_avg_acc)
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
            x_best, y_best, best_color, sign_best, autc_improv_best = x_plot, y_plot, color, sign, autc_improv
        else:
            # ax.plot(x_plot, y_plot, label=model_name, color=color, marker='.')
            if make_table:
                ax.plot(x_plot, y_plot, label='%s (%s%.1f%%)' % (model_name, sign, autc_improv), color=color)
            else:
                if model_name == 'no hist':
                    plt.plot(x_plot, y_plot, label='baseline', color=color)
                else:
                    plt.plot(x_plot, y_plot, label='%s (%s%.1f%%)' % (model_name, sign, autc_improv), color=color)
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
        label_best = 'best_u (%s%.1f%%)' % (sign_best, autc_improv_best)
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

            # for i in range(1, len(x)):
            #     if x[i] < x[i - 1]:
            #         x[i] = x[i - 1]
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
            # x = [min_x] + xs_seed[i]
            # y = ys_seed[i]
            # y = [y[0]] + y
            autc = autcs[i]
            if best_autc is None or autc > best_autc:
                best_autc = autc
                best_model = model_name
            if plot_tradeoffs:
                plt.plot(xs_seed[i], ys_seed[i], label='%s autc=%.5f' % (model_name, autc), color=color)
                # plt.plot(xs_seed[i], ys_seed[i], label='%s autc=%.5f' % (model_name, autc), marker='.', color=color)
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


def summarize(log_dir, log_set, user_name=''):
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

        h1_area = (x[-1] - x[0]) * h1_avg_acc
        autc_avg = auc(x, y) - h1_area
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

            h1_area = (x[-1] - x[0]) * h1_avg_acc
            autc_seed = auc(x, y) - h1_area
            if model_name == 'no hist':
                no_hist_autc_seeds.append(autc_seed)
            autcs_seed[-1].append(autc_seed)

    # avg
    autc_improvs_avg = [[] for i in range(len(model_names))]
    for i in range(len(model_names)):
        autc_improv_avg = autcs_avg[i] / no_hist_autc_avg - 1
        autc_improvs_avg[i].append(autc_improv_avg)
    # std
    autcs_seed, no_hist_autc_seeds = np.array(autcs_seed), np.array(no_hist_autc_seeds)
    autc_improvs_std = np.std(autcs_seed.T / no_hist_autc_seeds - 1, axis=1)
    autc_improvs_std = autc_improvs_std.reshape(len(autc_improvs_std), 1).tolist()

    # return [0], autc_improvs_by_seed, model_names
    return autc_improvs_avg, autc_improvs_std, model_names


def add_best_model(log_dir, valid_set, test_set):
    df_best = pd.read_csv('%s/best_models_%s.csv' % (log_dir, valid_set))
    df_test = pd.read_csv('%s/%s_log.csv' % (log_dir, test_set))
    groups_by_user = df_test.groupby('user')
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
            new_model_x.extend(df_seed['%s x' % best_model].tolist())
            new_model_y.extend(df_seed['%s y' % best_model].tolist())
            # print('(%s=%s) (%d=%d)' % (user_name, best_user, seed, best_seed))
            if user_name != best_user or seed != best_seed:
                raise ValueError('results and best lists not in same order of user -> seed')
            i += 1

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

    for user_idx in range(len(user_names)):
        user_name = user_names[user_idx]
        print('%d/%d user=%s' % (user_idx + 1, len(user_names), user_name))
        df_user = groups_by_user.get_group(user_name)
        user_len = df_user['len'].iloc[0]
        user_name_repeated = [user_name] * (num_bins + 1)
        user_len_repeated = [user_len] * (num_bins + 1)
        if seeds is None:
            seeds = pd.unique(df_user['seed'])
        groups_by_seed = df_user.groupby('seed')

        for seed_idx in seeds:
            seed = seeds[seed_idx]
            print('\t%d/%d seed=%d' % (seed_idx + 1, len(seeds), seed))
            seed_repeated = [seed] * (num_bins + 1)
            df_seed = groups_by_seed.get_group(seed)
            if inner_seeds is None:
                inner_seeds = pd.unique(df_seed['inner_seed'])
            groups_by_inner_seed = df_seed.groupby('inner_seed')

            for inner_seed_idx in inner_seeds:
                inner_seed = inner_seeds[inner_seed_idx]
                # print('\t\t%d/%d inner_seed=%d' % (inner_seed_idx + 1, len(inner_seeds), inner_seed))
                df_inner_seed = groups_by_inner_seed.get_group(inner_seed)
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
                            # y_bin = min(y_left, y_right)
                            y_bin = max(y_left, y_right)
                        else:
                            slope = (y_right - y_left) / (x_right - x_left)
                            y_bin = y_left + slope * (x_bin - x_left)
                        y_bins.append(y_bin)
                    dict_binarized['%s x' % model_name].extend(x_bins)
                    dict_binarized['%s y' % model_name].extend(y_bins)
    pd.DataFrame(dict_binarized).to_csv('%s/%s_bins_log.csv' % (log_dir, log_set), index=False)

    # print('binarizing...')
    #     df_results = pd.read_csv('%s/%s_log.csv' % (log_dir, log_set))
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
    #     df_bins.to_csv('%s/%s_bins_log.csv' % (log_dir, log_set), index=False)


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


def execute_phase(phase):
    binarize_by_compat = False
    individual_users = False
    get_best = False
    add_best = False
    make_summary = False

    print('\n%s' % phase)
    if phase == 'phase 1 - binarize validation results':
        log_set = 'valid'
        binarize_by_compat = True
    elif phase == 'phase 2 - get best_u for each user using binarized validation results':
        log_set = 'valid_bins'
        individual_users = True
        get_best = True
    elif phase == 'phase 3 - binarize test results':
        log_set = 'test'
        binarize_by_compat = True
    elif phase == 'phase 4 - add best_u computed from validation to binarized test results':
        log_set = 'valid_bins'
        add_best = True
    elif phase == 'phase 5 - generate averaged plots for binarized test results with best':
        log_set = 'test_bins_with_best'
    elif phase == 'phase 6 - generate individual user plots for test results':
        log_set = 'test_bins_with_best'
        individual_users = True
    elif phase == 'phase 7 - create test summary':
        log_set = 'test_bins_with_best'
        make_summary = True
        individual_users = True

    elif phase == 'generate user plots for binarized validation results':
        log_set = 'valid_bins'
        individual_users = True
    elif phase == 'generate averaged plots for binarized validation results':
        log_set = 'valid_bins'


    # print('phase 1 - for synthetic')
    # log_set = 'valid_bins'
    # individual_users = True
    # get_best = True

    # print('phase 2 - for synthetic')
    # # log_set = 'valid'
    # log_set = 'valid_bins'

    # default
    test_set = 'test_bins'
    count_best = False

    results_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/results/%s' % model_type
    log_dir = '%s/%s/%s/%s' % (results_dir, dataset, version, user_type)
    models = get_model_dict('jet')
    # models = get_model_dict('gist_rainbow')

    if add_best:
        add_best_model(log_dir, log_set, test_set)
    elif count_best:
        best_count_values(log_dir, log_set)
    elif binarize_by_compat:
        binarize_results_by_compat_values(log_dir, log_set, num_normalization_bins)
    elif not individual_users:
        if get_best:
            # best_model = get_best_models(log_dir, models, log_set)
            print('got best models for general set, not individual users!')
        else:
            plot_results(log_dir, dataset, user_type, models, log_set, bin_size=bin_size, show_tradeoff_plots=True)
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
                seeds, best_models_by_seed = get_best_models('%s/users_%s/logs' % (log_dir, log_set), models, log_set,
                                                             user_name=user_id, plot_tradeoffs=True)
                user_col += [user_id] * len(seeds)
                seed_col += seeds
                model_col += best_models_by_seed
            df = pd.DataFrame({'user': user_col, 'seed': seed_col, 'model': model_col})
            df.to_csv('%s/best_models_%s.csv' % (log_dir, log_set), index=False)
        elif make_summary:
            make_data_analysis(log_dir, dataset, user_type, target_col)
            user_col = []
            len_col = []
            weighted_distances = []
            # seed_col = []
            autc_improv_by_model_avg = None
            autc_improv_by_model_std = None
            wasserstein_distances = pd.read_csv('%s/wasserstein_distances.csv' % log_dir, index_col='user')
            feature_importances = pd.read_csv('%s/feature_importances.csv' % log_dir, index_col='user')
            gen_feature_importance = feature_importances.loc['general']
            for user_idx in range(len(user_ids)):
                user_id = user_ids[user_idx]
                hist_len = lens[user_idx]
                print('%d/%d user %s' % (user_idx + 1, len(user_ids), user_id))
                # seeds, autc_improv_by_model_user, model_names = summarize('%s/users_%s/logs' % (log_dir, log_set),
                #                                                           log_set,
                #                                                           user_name=user_id)
                autc_improv_by_model_user_avg, autc_improv_by_model_user_std, model_names = summarize(
                    '%s/users_%s/logs' % (log_dir, log_set), log_set, user_name=user_id)
                user_col += [user_id]  # * len(seeds)
                len_col += [hist_len]  # * len(seeds)
                # weighted_distances += [np.average(wasserstein_distances.loc[user_id],
                #                       weights=gen_feature_importance)] * len(seeds)
                weighted_distances += [np.average(wasserstein_distances.loc[user_id], weights=gen_feature_importance)]
                # seed_col += seeds
                if autc_improv_by_model_avg is None:
                    autc_improv_by_model_avg = autc_improv_by_model_user_avg
                    autc_improv_by_model_std = autc_improv_by_model_user_std
                else:
                    for i in range(len(autc_improv_by_model_avg)):
                        autc_improv_by_model_avg[i].extend(autc_improv_by_model_user_avg[i])
                        autc_improv_by_model_std[i].extend(autc_improv_by_model_user_std[i])

            # all users together
            autc_by_model_all_users_avg, autc_by_model_all_users_std, model_names = summarize(log_dir, log_set)
            user_col += ['all users']
            len_col += [sum(len_col)]
            weighted_distances += [0]
            for i in range(len(autc_improv_by_model_avg)):
                autc_improv_by_model_avg[i].extend(autc_by_model_all_users_avg[i])
                autc_improv_by_model_std[i].extend(autc_by_model_all_users_std[i])

            # df_dict = {'user': user_col, 'len': len_col, 'dist': weighted_distances, 'seed': seed_col}
            for metric, metric_data in [['avg', autc_improv_by_model_avg], ['std', autc_improv_by_model_std]]:
                df_dict = {'user': user_col, 'len': len_col, 'dist': weighted_distances}
                for i in range(len(model_names)):
                    df_dict[model_names[i]] = metric_data[i]
                df = pd.DataFrame(df_dict)
                df.to_csv('%s/summary_of_%s_%s.csv' % (log_dir, log_set, metric), index=False)
        else:
            for user_idx in range(len(user_ids)):
                user_id = user_ids[user_idx]
                print('%d/%d user=%s' % (user_idx + 1, len(user_ids), user_id))
                plot_results('%s/users_%s' % (log_dir, log_set), dataset, user_type, models, log_set,
                             bin_size=bin_size, user_name=user_id)


# dataset = 'ednet'
# version = 'unbalanced/20 users'
# user_type = 'user'
# target_col = 'correct_answer'
# model_type = 'simulated annealing'
# bin_size = 10

# dataset = 'assistment'
# # version = 'unbalanced/inner seeds'
# # version = 'unbalanced/large val'
# version = 'unbalanced/many seeds'
# user_type = 'user_id'
# target_col = 'correct'
# model_type = 'simulated annealing'
# bin_size = 10

# dataset = 'salaries'
# version = 'unbalanced/inner seeds'
# user_type = 'relationship'
# target_col = 'salary'
# model_type = 'simulated annealing'
# bin_size = 10

dataset = 'recividism'
version = 'unbalanced/short'
user_type = 'race'
target_col = 'is_recid'
model_type = 'simulated annealing'
bin_size = 10

# dataset = 'averaging tradeoffs'
# version = 'justifying stdev of delta'
# user_type = 'synthetic_user'
# model_type = 'synthetic'
# bin_size = 1

num_normalization_bins = 100

# todo: CHOOSE EXPERIMENT PHASE
phases = [
    'phase 1 - binarize validation results',
    # 'phase 2 - get best_u for each user using binarized validation results',
    # 'phase 3 - binarize test results',
    # 'phase 4 - add best_u computed from validation to binarized test results',
    # 'phase 5 - generate averaged plots for binarized test results with best',
    # 'phase 6 - generate individual user plots for test results',
    # 'phase 7 - create test summary',
    # 'generate user plots for binarized validation results',
    'generate averaged plots for binarized validation results'
]

for phase in phases:
    execute_phase(phase)

print('\ndone')
