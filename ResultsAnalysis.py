import csv
import numpy as np
import os
from sklearn.metrics import auc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)


def weighted_cov(x, y, w):
    return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)


def weighted_corr(x, y, w):
    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))


def make_monotonic(x, y):
    i = 0
    while i < len(x) - 1:
        if x[i + 1] < x[i]:
            del x[i + 1]
            del y[i + 1]
        else:
            i += 1


def cutoff(x, y, cutoff_x):
    i = len(x) - 1
    while i >= 0:
        if x[i] > cutoff_x:
            del x[i]
            del y[i]
            i -= 1
        else:
            break


def auc_summary(auc_path, results_path):
    df_auc = pd.read_csv(auc_path).groupby('user_id').mean()
    df_auc_summary = pd.DataFrame({'metric': ['auc', 'improv']})
    for col in df_auc:
        if 'no hist' in col:
            no_hist_avg = np.average(df_auc[col], weights=df_auc['instances'])
            break
    for col in df_auc:
        if 'area' in col:
            col_avg = np.average(df_auc[col], weights=df_auc['instances'])
            improv = col_avg / no_hist_avg - 1
            df_auc_summary[col] = [col_avg, improv]
    df_auc_summary.to_csv(results_path, index=False)


def area_calculator(log_path, hybrid_log_path, results_path, model_names, cut_by_min=False):
    log_df = pd.read_csv(log_path)
    hybrid_log_df = pd.read_csv(hybrid_log_path)

    log_by_users = log_df.groupby(['user_id'])
    hybrid_log_by_users = hybrid_log_df.groupby(['user_id'])
    user_ids = log_df.user_id.unique()
    seeds = log_df['train seed'].unique()

    with open(results_path, 'w', newline='') as file_out:
        header = ['train frac',
                  'user_id',
                  'instances',
                  'train seed',
                  'comp range',
                  'acc range',
                  'h1 acc']
        for model_name in model_names:
            header += [model_name + ' area']
        writer = csv.writer(file_out)
        writer.writerow(header)

        user_count = 0
        for user_id in user_ids:
            user_count += 1
            print(str(user_count) + '/' + str(len(user_ids)) + ' user ' + str(user_id))

            user_log = log_by_users.get_group(user_id).reset_index(drop=True)
            user_hybrid_log = hybrid_log_by_users.get_group(user_id).reset_index(drop=True)

            # user_log = log_by_users.get_group(user_id).groupby('diss weight').mean().reset_index(drop=True)
            # user_hybrid_log = hybrid_log_by_users.get_group(user_id).groupby('std offset').mean().reset_index(drop=True)

            for seed in seeds:
                first_row_of_user = user_log.loc[0]
                history_len = first_row_of_user['instances']
                # seed = first_row_of_user['train seed']
                com_range = first_row_of_user['comp range']
                auc_range = first_row_of_user['acc range']
                h1_acc = first_row_of_user['h1 acc']

                # sim = '%.4f' % (user_hybrid_stat_log.loc[0]['sim'],)

                row = [
                    first_row_of_user['train frac'],
                    str(user_id),
                    history_len,
                    # sim,
                    seed,
                    com_range,
                    auc_range,
                    h1_acc]

                models_x = []
                models_y = []
                for model_name in model_names:
                    if 'hybrid' not in model_name:
                        log = user_log
                    else:
                        log = user_hybrid_log
                    log = log.loc[log['train seed'] == seed]
                    models_x += [log[model_name + ' x'].tolist()]
                    models_y += [log[model_name + ' y'].tolist()]

                min_x = min(min(i) for i in models_x)

                mono_xs = [[min_x] + i.copy() for i in models_x]
                mono_ys = [[i[0]] + i.copy() for i in models_y]

                for i in range(len(mono_xs)):
                    make_monotonic(mono_xs[i], mono_ys[i])

                if cut_by_min:
                    cutoff_x = 1
                    for mono_x in mono_xs:
                        max_mono_x = max(mono_x)
                        cutoff_x = min(cutoff_x, max_mono_x)

                    for i in range(len(mono_xs)):
                        cutoff(mono_xs[i], mono_ys[i], cutoff_x)

                h1_area = (1 - min_x) * h1_acc

                areas = [auc(mono_xs[i] + [mono_xs[i][-1], 1], mono_ys[i] + [h1_acc, h1_acc]) - h1_area
                         for i in range(len(mono_xs))]
                writer.writerow(row + areas)

    auc_summary(results_path, results_path.replace('auc', 'auc_summary'))

    # h1_x = [min_x, 1]
    # h1_y = [h1_acc, h1_acc]
    # plt.plot(h1_x, h1_y, 'k--', marker='.', label='h1')
    # plt.text(min_x, h1_acc * 1.005, 'h1')

    # if not skip_L_models:
    #     if not only_L1:
    #         plt.plot(no_hist_x, no_hist_y, 'b', marker='.', linewidth=3, markersize=22,
    #                  label='h2 not using history')
    #         plt.plot(hybrid_stat_x, hybrid_stat_y, 'g', marker='.', linewidth=3, markersize=18,
    #                  label='h2 hybrid stat')
    #         plt.plot(hybrid_nn_x, hybrid_nn_y, 'seagreen', marker='.', linewidth=3, markersize=18,
    #                  label='h2 hybrid nn')
    #         plt.plot(L0_x, L0_y, 'r', marker='.', linewidth=3, markersize=14,
    #                  label='h2 using L0')
    #         plt.plot(L1_x, L1_y, 'm', marker='.', linewidth=3, markersize=10,
    #                  label='h2 using L1')
    #         plt.plot(L2_x, L2_y, 'orange', marker='.', linewidth=3, markersize=6,
    #                  label='h2 using L2')
    #     else:
    #         plt.plot(no_hist_x, no_hist_y, 'b', marker='.', linewidth=3, markersize=18,
    #                  label='h2 not using history')
    #         plt.plot(hybrid_stat_x, hybrid_stat_y, 'g', marker='.', linewidth=3, markersize=16,
    #                  label='h2 hybrid stat')
    #         plt.plot(hybrid_nn_x, hybrid_nn_y, 'seagreen', marker='.', linewidth=3, markersize=14,
    #                  label='h2 hybrid nn')
    #         plt.plot(L1_x, L1_y, 'm', marker='.', linewidth=3, markersize=12,
    #                  label='h2 using L1')
    # else:
    #     plt.plot(no_hist_x, no_hist_y, 'b', marker='.', linewidth=3, markersize=18,
    #              label='h2 not using history')
    #     plt.plot(hybrid_stat_x, hybrid_stat_y, 'g', marker='.', linewidth=3, markersize=16,
    #              label='h2 hybrid stat')
    #     plt.plot(hybrid_nn_x, hybrid_nn_y, 'seagreen', marker='.', linewidth=3, markersize=14,
    #              label='h2 hybrid nn')
    #
    # plt.xlabel('compatibility')
    # plt.ylabel('accuracy')
    # # plt.legend()
    #
    # # plt.title(
    # #     'user=' + str(user_id) + ' hist_len=' + str(history_len) + ' split=' + str(x['train frac'])
    # #     + ' sim=' + sim + '\n\n')
    #
    # plt.title(
    #     'user=' + str(user_id) + ' hist_len=' + str(int(history_len)) + ' split=' + str(first_row_of_user['train frac']) + '\n\n')
    #
    # areas = [no_hist_area, hybrid_stat_area, hybrid_nn_area]
    # col_labels = ['no hist', 'hybrid stat', 'hybrid nn']
    # colors = ['b', 'g', 'seagreen']
    # if not skip_L_models:
    #     if not only_L1:
    #         areas = [no_hist_area, L0_area, L1_area, L2_area, hybrid_stat_area, hybrid_nn_area]
    #         col_labels = ['no hist', 'L0', 'L1', 'L2', 'hybrid stat', 'hybrid nn']
    #         colors = ['b', 'r', 'm', 'orange', 'g', 'seagreen']
    #     else:
    #         areas = [no_hist_area, L1_area, hybrid_stat_area, hybrid_nn_area]
    #         col_labels = ['no hist', 'L1', 'hybrid stat', 'hybrid nn']
    #         colors = ['b', 'm', 'g', 'seagreen']
    #
    #
    # cell_text = [(['%1.4f' % (area) for area in areas])]
    # row_labels = ['area']
    #
    # table = plt.table(cellText=cell_text, rowLabels=row_labels, colColours=colors, colLabels=col_labels,
    #                   loc='top', cellLoc='center')
    # table.scale(1, 1.2)
    # plt.subplots_adjust(top=0.85)
    #
    # xs = [mono_no_hist_x, mono_hybrid_stat_x, mono_hybrid_nn_x]
    # ys = [mono_no_hist_y, mono_hybrid_stat_y, mono_hybrid_nn_y]
    # if not skip_L_models:
    #     if not only_L1:
    #         xs = [mono_no_hist_x, mono_L0_x, mono_L1_x, mono_L2_x, mono_hybrid_stat_x, mono_hybrid_nn_x]
    #         ys = [mono_no_hist_y, mono_L0_y, mono_L1_y, mono_L2_y, mono_hybrid_stat_y, mono_hybrid_nn_y]
    #     else:
    #         xs = [mono_no_hist_x, mono_L1_x, mono_hybrid_stat_x, mono_hybrid_nn_x]
    #         ys = [mono_no_hist_y, mono_L1_y, mono_hybrid_stat_y, mono_hybrid_nn_y]
    #
    #
    # for i in range(len(xs)):
    #     first_row_of_user = xs[i]
    #     y = ys[i]
    #     plt.fill_between(first_row_of_user, [h1_acc] * len(first_row_of_user), y, facecolor=colors[i], alpha=0.2)
    #
    # # plt.savefig(
    # #     plots_dir + '\\by_hist_length\\len_' + str(history_len) + '_student_' + str(
    # #         user_id) + '.png')
    # # plt.savefig(
    # #     plots_dir + '\\by_accuracy_range\\acc_' + '%.4f' % (auc_range,) + '_student_' + str(
    # #         user_id) + '.png')
    # # plt.savefig(
    # #     plots_dir + '\\by_compatibility_range\\com_' + '%.4f' % (
    # #     com_range,) + '_student_' + str(
    # #         user_id) + '.png')
    # # plt.savefig(
    # #     plots_dir + '\\by_user_id\\student_' + str(user_id) + '_seed_' + str(seed) + '.png')
    #
    # plt.savefig(
    #     results_path + '\\plots\\student_' + str(user_id) + '.png')
    # # plt.show()
    # plt.clf()


def safe_auc(x, y):
    x_reset = x.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    first_index = y_reset.first_valid_index()
    x_reset = list(x_reset[first_index:])
    y_reset = list(y_reset[first_index:])
    # make_monotonic(x_reset, y_reset)
    return auc(x_reset, y_reset)


def get_fixed_std(df_model, h1_mean_acc, users_h1_accs, weighted=False):
    users = df_model.groupby('user')
    model_mean_initial_acc = df_model.groupby('position').mean().reset_index(drop=True)['y'][0]
    df_fixed = pd.DataFrame(columns=['position', 'y'])
    for user_id, user_data in users:
        user_data = user_data.reset_index(drop=True)

        if weighted:
            size = int(user_data['size'][0])
            user_data = user_data.drop(columns=['size'])
            user_data = pd.concat([user_data] * size, ignore_index=True)

        user_y = user_data['y']
        model_user_initial_acc = user_y[0]
        h1_user_acc = users_h1_accs[user_id]
        y_min = min(min(user_y), h1_user_acc)
        y_max = max(max(user_y), h1_user_acc)
        if y_max == y_min:
            # if h1_user_acc >= model_user_initial_acc or y_max == y_min:
            continue
        fixed_y = (user_data['y'] - model_user_initial_acc) * (
                (model_mean_initial_acc - h1_mean_acc) / (y_max - y_min)) + model_mean_initial_acc
        df_fixed = df_fixed.append(pd.DataFrame({'position': user_data['position'], 'y': fixed_y}))
    return df_fixed.groupby('position').std()['y']


def plots_averaged_over_all_users(log_path, hybrid_log_path, results_dir, model_names, skip_users, user_type='users',
                                  individual_user_id='', simple_plots=False, weighted=True, plot_std=True, alpha=0.2):
    labels_dict = {
        'no hist': 'no history',
        'L0': 'old L0',
        'L1': 'old L1',
        'L2': 'old L2',
        'L3': 'L0',
        'L4': 'L1',
        'hybrid': 'hybrid',
        'full_hybrid': 'full_hybrid',
        'baseline': 'baseline',
        'adaboost': 'adaboost',
        'comp_adaboost': 'comp_adaboost',
    }
    colors_dict = {
        'no hist': 'k',
        'L0': 'b',
        'L1': 'purple',
        'L2': 'orange',
        'L3': 'r',
        'L4': 'b',
        'hybrid': 'g',
        'full_hybrid': 'c',
        'baseline': 'grey',
        'adaboost': 'purple',
        'comp_adaboost': 'orange',
    }
    markers_dict = {
        'no hist': 'o',
        'L0': 'v',
        'L1': '^',
        'L2': '<',
        'L3': '>',
        'L4': '^',
        'hybrid': 'D',
        'full_hybrid': 's',
        'baseline': '.',
        'adaboost': '<',
        'comp_adaboost': 'v',
    }
    markersizes_dict = {
        'no hist': 4,
        'L0': 6,
        'L1': 6,
        'L2': 6,
        'L3': 6,
        'L4': 6,
        'hybrid': 4,
        'full_hybrid': 4,
        'baseline': 4,
        'adaboost': 6,
        'comp_adaboost': 6,
    }
    labels = [labels_dict[i] for i in model_names]
    colors = [colors_dict[i] for i in model_names]
    markers = [markers_dict[i] for i in model_names]
    markersizes = [markersizes_dict[i] for i in model_names]

    log_df = pd.read_csv(log_path)
    if 'hybrid' in model_names:
        hybrid_log_df = pd.read_csv(hybrid_log_path)

    log_by_users = log_df.groupby(['user_id'])
    if 'hybrid' in model_names:
        hybrid_log_by_users = hybrid_log_df.groupby(['user_id'])

    user_ids = log_df.user_id.unique()

    final_df = pd.DataFrame(columns=['model', 'user', 'size', 'position', 'x', 'y'])

    user_count = 0
    for user_id in user_ids:
        if user_id in skip_users:
            continue
        user_count += 1
        print(str(user_count) + '/' + str(len(user_ids)) + ' user ' + str(user_id))

        user_log = log_by_users.get_group(user_id).reset_index(drop=True).groupby('diss weight').mean()
        if 'hybrid' in model_names:
            user_hybrid_log = hybrid_log_by_users.get_group(user_id).reset_index(drop=True).groupby('std offset').mean()

        size = user_log['instances'][0]

        merged_df = pd.DataFrame(columns=['model', 'user', 'size', 'position', 'x', 'y'])
        merged_df = merged_df.append(
            pd.DataFrame(
                {'model': 'h1', 'user': user_id, 'size': size, 'position': 0, 'x': 0.0,
                 'y': [user_log.loc[0]['h1 acc']]}))
        for model_name in model_names:
            log = user_log
            positions = range(len(log))
            # if model_name == 'hybrid':
            if 'hybrid' in model_name:
                log = user_hybrid_log
                positions = reversed(range(len(log)))
            merged_df = merged_df.append(
                pd.DataFrame({'model': model_name, 'user': user_id, 'size': size, 'position': positions,
                              'x': log[model_name + ' x'], 'y': log[model_name + ' y']}))
        final_df = final_df.append(merged_df)

    groups_by_model = final_df.groupby('model')
    h1_group = groups_by_model.get_group('h1')[['size', 'y']].reset_index(drop=True)
    h1_acc_mean = np.average(h1_group['y'], weights=h1_group['size'])

    models_x = []
    models_y = []
    for model_name in model_names:
        group = groups_by_model.get_group(model_name)

        if weighted:
            weighted_group = group.copy()
            weighted_group['y'] = weighted_group['y'] * weighted_group['size']
            weighted_group['x'] = weighted_group['x'] * weighted_group['size']
            weighted_sum = weighted_group.groupby('position').sum()
            x = list(weighted_sum['x'] / weighted_sum['size'])
            y = list(weighted_sum['y'] / weighted_sum['size'])
        else:
            group_mean = group.groupby('position').mean()
            x = list(group_mean['x'])
            y = list(group_mean['y'])

        models_x += [x]
        models_y += [y]

    min_x = min(min(i) for i in models_x)
    max_x = max(max(i) for i in models_x)

    h1_x = [min_x, max_x]
    h1_y = [h1_acc_mean, h1_acc_mean]

    linewidth = 2
    h1_marker_size = 6
    marker_delta = 0

    if simple_plots:
        linewidth = 3
        h1_marker_size = 18
        marker_delta = 6
        matplotlib.rcParams.update({'font.size': 22})
        # plt.axis('off')

    for i in range(len(model_names)):
        model_name = model_names[i]
        if model_name != 'baseline':
            plt.plot(models_x[i], models_y[i], colors[i], marker=markers[i], markersize=markersizes[i] + marker_delta,
                     label=labels[i], linewidth=linewidth)
        else:
            plt.plot(models_x[i], models_y[i], colors[i], label=labels[i], linewidth=linewidth,
                     dashes=[5, 1.5, 1.5, 1.5])
            plt.plot([models_x[i][-1]], [models_y[i][-1]], colors[i], marker='s',
                     markersize=markersizes[i] + marker_delta)

    plt.plot(h1_x, h1_y, 'k', marker='.', linewidth=linewidth, markersize=h1_marker_size, label='pre-update',
             dashes=[3, 1.5])

    if not simple_plots:
        xlabel = 'compatibility'
        ylabel = 'average accuracy'
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='lower left')
        plt.grid()

    if plot_std:
        users_h1_accs = {}
        for user_id, user_data in groups_by_model.get_group('h1').groupby('user'):
            user_data = user_data.reset_index(drop=True)
            users_h1_accs[user_id] = user_data['y'][0]

        stds = [get_fixed_std(groups_by_model.get_group(i), h1_acc_mean, users_h1_accs, weighted) for i in model_names]
        for i in range(len(models_x)):
            if model_names[i] == 'baseline':
                continue
            x = models_x[i]
            y = models_y[i]
            std = stds[i]
            plt.fill_between(x, y + std, y - std, facecolor=colors[i], alpha=alpha)

    if individual_user_id == '':
        title = 'average plots for users = %s' % user_type
        file_name = 'average_plot_%s' % user_type
    else:
        title = '%s = %s, n = %d' % (user_type, individual_user_id, size)
        file_name = 'len_%d_user_%s' % (size, individual_user_id)
    if not simple_plots:
        plt.title(title)
    plt.savefig(results_dir + '\\' + file_name + '.png', bbox_inches=0)
    plt.show()
    plt.clf()


def split_users(log_path, hybrid_log_path, results_dir):
    df_log = pd.read_csv(log_path)
    df_hybrid_log = pd.read_csv(hybrid_log_path)
    log_by_users = df_log.groupby('user_id')
    hybrid_log_by_users = df_hybrid_log.groupby('user_id')

    for user_id, user_data in log_by_users:
        mod_user_id = []
        for row in user_data['train seed']:
            mod_user_id += [str(user_id) + '_' + str(row)]
        user_data['user_id'] = mod_user_id
        user_data.to_csv(results_dir + '%s_log.csv' % user_id, index=False)
    for user_id, user_data in hybrid_log_by_users:
        mod_user_id = []
        for row in user_data['train seed']:
            mod_user_id += [str(user_id) + '_' + str(row)]
        user_data['user_id'] = mod_user_id
        user_data.to_csv(results_dir + '%s_hybrid_log.csv' % user_id, index=False)


def correlate_baseline_with_models(log_path, results_dir):
    df_log = pd.read_csv(log_path)
    log_by_users = df_log.groupby('user_id')
    header = ['user_id', 'hist_len', 'seed', 'diss weight', 'h1 acc', 'baseline delta']
    models = []
    for col in df_log.drop(columns=['no hist x', 'no hist y', 'baseline x', 'baseline y']):
        if ' x' in col:
            model = col[:-2]
            models += [model]
            header += [
                # model + ' acc',
                model + ' delta',
                ]
    df_result = pd.DataFrame(columns=header)
    diss_weights = df_log['diss weight'].unique()
    for user_id, user_log in log_by_users:
        user_log = user_log.reset_index(drop=True)
        user_log_by_seed = user_log.groupby('train seed')
        hist_len = user_log.loc[0]['instances']
        for seed, user_seed_log in user_log_by_seed:
            user_seed_log = user_seed_log.reset_index(drop=True)
            h1_y = user_seed_log['h1 acc'][0]
            baseline_y = user_seed_log['baseline y'].reset_index(drop=True)
            baseline_y_delta = baseline_y - baseline_y[0]  # delta from point where lambda=0

            # extend no hist plot to min and max values of x
            no_hist_x = list(user_seed_log['no hist x'])
            no_hist_y = list(user_seed_log['no hist y'])
            make_monotonic(no_hist_x, no_hist_y)
            no_hist_x = [0] + no_hist_x + [no_hist_x[-1], 1]
            no_hist_y = [no_hist_y[0]] + no_hist_y + [h1_y, h1_y]

            # baseline_y_delta = (baseline_y_delta - np.min(baseline_y_delta))/np.ptp(baseline_y_delta)

            model_rows = None
            df_user_seed = pd.DataFrame({'user_id': user_id,
                                         'hist_len': hist_len,
                                         'seed': seed,
                                         'diss weight': diss_weights,
                                         'h1 acc': h1_y,
                                         'baseline delta': baseline_y_delta,
                                         # 'h2 acc': no_hist_adjusted_y,
                                         })
            for col in user_seed_log.drop(columns=['no hist x', 'no hist y', 'baseline x', 'baseline y']):
                if ' x' in col:
                    model = col[:-2]
                    model_x = user_seed_log[model + ' x']
                    model_y = user_seed_log[model + ' y']
                    no_hist_adjusted_y = pd.Series([get_y_given_x(no_hist_x, no_hist_y, x) for x in model_x])
                    model_y_delta = model_y - no_hist_adjusted_y  # delta from model y to no hist y in same x
                    df_user_seed[model + ' delta'] = model_y_delta

            df_result = df_result.append(df_user_seed)

    df_result.to_csv(results_dir+'corr.csv', index=False)
    baseline_delta = df_result['baseline delta']
    with open(results_dir+'corr.txt', 'w', newline='') as file_out:
        for model in models:
            model_delta = df_result[model + ' delta']
            pearson = pearsonr(baseline_delta, model_delta)
            spearman = spearmanr(baseline_delta, model_delta)
            file_out.write(model+': pearson='+str(pearson[0])+' p-val='+str(pearson[1])
                           + ', spearman='+str(spearman[0])+' p-val='+str(spearman[1]) + '\n')


def get_y_given_x(X, Y, x):
    i = 0
    while not X[i] <= x <= X[i + 1]:
        i += 1
    x_left = X[i]
    y_left = Y[i]
    x_right = X[i + 1]
    y_right = Y[i + 1]
    if x_left == x_right:  # vertical line
        return max(y_left, y_right)
    slope = (y_right - y_left) / (x_right - x_left)
    return y_left + slope * (x - x_left)


skip_users = []

dataset_name = 'salaries'
version = '1'
user_types = ['relationship']

# dataset_name = 'assistment'
# version = 'layers []'
# user_types = ['user_id']

# dataset = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\recividism\\'
# version = '80 split [10] h1 500 h2 200 epochs\\'
# user_types = ['race']
# # user_types = ['race', 'sex', 'age_cat', 'c_charge_degree', 'score_text']

# dataset = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\titanic\\'
# version = '80 split [] epochs h1 500 h2 800\\'
# user_types = ['Pclass', 'Sex', 'AgeClass', 'Embarked']

# dataset_name = 'mooc'
# version = '80 split []\\'
# user_types = ['forum_uid']

# dataset_name = 'abalone'
# version = '80 split [5]\\'
# user_types = ['sex']

# dataset_name = 'hospital_mortality'
# version = '80 split []\\'
# user_types = ['MARITAL_STATUS']

# analysis settings
individual_users = True
compute_auc = False
correlation = False

# plot settings
simple_plots = False
weighted = True
plot_std = True
opacity = 0.15

dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\decision trees\\%s\\' % dataset_name

# model_names = version.split('-')

model_names = [
    'no hist',
    # 'L0',
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

version += '\\'
for user_type in user_types:
    user_type_dir = dataset_path + version + user_type + '\\'
    log_path = user_type_dir + 'log.csv'
    hybrid_log_path = user_type_dir + 'hybrid_log.csv'

    if compute_auc:
        area_calculator(log_path, hybrid_log_path, '%sauc.csv' % user_type_dir, model_names)
        continue

    if correlation:
        correlate_baseline_with_models(log_path, user_type_dir)
        continue

    if not individual_users:
        results_dir = dataset_path + version + 'averaged plots\\by user type'
        safe_make_dir(results_dir)
        plots_averaged_over_all_users(log_path, hybrid_log_path, results_dir, model_names, skip_users,
                                      user_type=user_type, simple_plots=simple_plots, weighted=weighted,
                                      plot_std=plot_std, alpha=opacity)
    else:
        results_dir = dataset_path + version + 'averaged plots\\by user\\%s' % user_type
        if not os.path.exists(results_dir):
            by_user_id_dir = '%s\\log_by_user\\' % user_type_dir
            safe_make_dir(by_user_id_dir)
            split_users(log_path, hybrid_log_path, by_user_id_dir)
        safe_make_dir(results_dir)
        user_ids = pd.unique(pd.read_csv(log_path)['user_id'])
        for user_id in user_ids:
            user_log_path = '%slog_by_user\\%s_log.csv' % (user_type_dir, user_id)
            user_hybrid_log_path = '%slog_by_user\\%s_hybrid_log.csv' % (user_type_dir, user_id)
            plots_averaged_over_all_users(user_log_path, user_hybrid_log_path, results_dir, model_names, skip_users,
                                          user_type=user_type, individual_user_id=user_id, simple_plots=simple_plots,
                                          weighted=weighted, plot_std=plot_std, alpha=opacity)
