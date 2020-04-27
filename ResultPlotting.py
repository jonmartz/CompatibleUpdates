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
        mod_user_id = []
        for row in user_data['seed']:
            mod_user_id += [str(user_id) + '_' + str(row)]
        user_data['user'] = mod_user_id
        user_data.to_csv('%s\\%s_user_logs\\%s_%s.csv' % (log_dir, log_set, log_set, user_id), index=False)


def get_model_dict(cmap_name):
    models = {
        'no diss': {'sample_weight': [1, 0, 1, 0], 'color': 'grey'},
        'no hist': {'sample_weight': [1, 1, 0, 0], 'color': 'black'},
        # 'sim_ann': {'sample_weight': [0.0, 0.6352, 0.3119, 0.0780], 'color': 'purple'},
        'hybrid': {'sample_weight': ['', '', '', ''], 'color': 'red'},
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


def plot_results(log_dir, dataset, user_type, models, log_set, bin_size=1, user_name='', show_tradeoff_plots=True,
                 smooth_color_progression=False):

    if user_name == '':
        df_results = pd.read_csv('%s\\%s_log.csv' % (log_dir, log_set))
    else:
        df_results = pd.read_csv('%s\\%s_%s.csv' % (log_dir, log_set, user_name))

    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    xs, ys, xs_plot, ys_plot = [], [], [], []
    autcs = []
    h1_avg_acc = np.average(df_results['h1_acc'], weights=df_results['len'])
    groups_by_weight = df_results.groupby('weight')

    fig, (ax, tabax) = plt.subplots(nrows=2, figsize=(6.4, 4.8 + 0.3 * len(model_names)))
    cmap = plt.cm.get_cmap('jet')

    for model_name in model_names:

        dfs_by_weight = [groups_by_weight.get_group(i) for i in groups_by_weight.groups]
        x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
        y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]

        # make x monotonic for AUTC
        x_copy = x.copy()
        for i in range(1, len(x_copy)):
            if x_copy[i] < x_copy[i - 1]:
                x_copy[i] = x_copy[i - 1]
        h1_area = (x_copy[-1] - x_copy[0]) * h1_avg_acc
        autc = auc(x_copy, y) - h1_area
        autcs.append(autc)
        if model_name == 'no hist':
            no_hist_autc = autc

        xs.append(x)
        ys.append(y)
        if bin_size > 1:
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
    for i in range(len(model_names)):
        x = xs[i]
        y = ys[i]
        autc = autcs[i]
        min_x_model = min(x)
        if min_x_model > min_x:  # for models that start at better compatibility
            autc += (min_x_model - min_x) * (y[0] - h1_avg_acc)
        autc_improvs.append((autc / no_hist_autc - 1) * 100)

    sorted_idxs = [autc_improvs.index(i) for i in reversed(sorted(autc_improvs))]

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

        ax.plot(x_plot, y_plot, marker='.', label=model_name, color=color)
        if model_name == 'no hist':
            color = 'white'
        colors.append(color)

    # table
    columns = ('gen', 'gen diss', 'hist', 'hist diss', '+ AUTC %')
    tabax.axis("off")
    tabax.table(cellText=cell_text,
                rowLabels=model_names_sorted,
                rowColours=colors,
                colLabels=columns,
                loc='center',
                )

    ax.set_xlabel('compatibility')
    ax.set_ylabel('accuracy')
    if user_name == '':
        title = '%s tradeoffs, dataset=%s user_type=%s' % (log_set, dataset, user_type)
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


dataset = 'assistment'
version = 'unbalanced\\without skills'
user_type = 'user_id'

# dataset = 'salaries'
# version = '1'
# user_type = 'relationship'

# log_set = 'test'
log_set = 'valid'
# individual_users = False
individual_users = True
bin_size = 2

results_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\simulated annealing'
log_dir = '%s\\%s\\%s\\%s' % (results_dir, dataset, version, user_type)
models = get_model_dict('jet')
# models = get_model_dict('gist_rainbow')

if not individual_users:
    plot_results(log_dir, dataset, user_type, models, log_set, bin_size=bin_size)
else:
    users_dir = '%s\\%s\\%s\\%s\\%s_user_logs' % (results_dir, dataset, version, user_type, log_set)
    if not os.path.exists(users_dir):
        safe_make_dir(users_dir)
        split_users(log_dir, log_set)
    user_ids = pd.unique(pd.read_csv('%s\\%s_log.csv' % (log_dir, log_set))['user'])
    for user_id in user_ids:
        plot_results('%s\\%s_user_logs' % (log_dir, log_set), dataset, user_type, models, log_set, bin_size=bin_size,
                     user_name=user_id)

