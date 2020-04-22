import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def safe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_users(log_dir):
    df_log = pd.read_csv('%s\\log.csv' % log_dir)
    log_by_users = df_log.groupby('user')

    for user_id, user_data in log_by_users:
        mod_user_id = []
        for row in user_data['seed']:
            mod_user_id += [str(user_id) + '_' + str(row)]
        user_data['user'] = mod_user_id
        user_data.to_csv('%s\\user_logs\\%s.csv' % (log_dir, user_id), index=False)


def get_model_dict():
    models = {}
    skip_models = [
        [1, 1, 0, 0],  # no hist
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
    ]
    cmap = plt.cm.get_cmap('jet')
    models['no hist'] = {'sample_weight': [1, 1, 0, 0], 'color': 'black'}
    model_name = 1
    for i0 in [0, 1]:
        for i1 in [0, 1]:
            for i2 in [0, 1]:
                for i3 in [0, 1]:
                    name = 'L%d' % model_name
                    sample_weight = [i0, i1, i2, i3]
                    if sample_weight in skip_models:
                        continue
                    models[name] = {'sample_weight': sample_weight,
                                    'color': cmap(model_name / (16 - len(skip_models) + 2))}
                    model_name += 1
    best_sample_weight = [0.0, 0.6352316047435935, 0.3119101971209735, 0.07805665820394585]
    models['sim_ann'] = {'sample_weight': best_sample_weight, 'color': cmap(1.0)}
    models['hybrid'] = {'sample_weight': ['', '', '', ''], 'color': 'green'}
    return models


def plot_results(log_dir, dataset, user_type, models, bin_size=1, user_name='', show_tradeoff_plots=True):

    if user_name == '':
        df_results = pd.read_csv('%s\\log.csv' % log_dir)
    else:
        df_results = pd.read_csv('%s\\%s.csv' % (log_dir, user_name))

    model_names = [i[:-2] for i in df_results.columns if ' x' in i]
    xs = []
    ys = []
    autcs = []
    h1_avg_acc = np.average(df_results['h1_acc'], weights=df_results['len'])
    groups_by_weight = df_results.groupby('weight')

    fig, (ax, tabax) = plt.subplots(nrows=2, figsize=(6.4, 4.8 + 0.3 * len(model_names)))
    cmap = plt.cm.get_cmap('jet')

    for model_name in model_names:

        dfs_by_weight = [groups_by_weight.get_group(i) for i in groups_by_weight.groups]
        x = [np.average(i['%s x' % model_name], weights=i['len']) for i in dfs_by_weight]
        y = [np.average(i['%s y' % model_name], weights=i['len']) for i in dfs_by_weight]

        if bin_size > 1:
            last_idx = len(x) - ((len(x) - 2) % bin_size) - 1
            x_binned = np.mean(np.array(x[1:last_idx]).reshape(-1, bin_size), axis=1)
            y_binned = np.mean(np.array(y[1:last_idx]).reshape(-1, bin_size), axis=1)
            xs.append([x[0]] + list(x_binned) + [np.mean(x[last_idx:-1]), x[-1]])
            ys.append([y[0]] + list(y_binned) + [np.mean(y[last_idx:-1]), y[-1]])
        else:
            xs.append(x.copy())
            ys.append(y)

        # make x monotonic for AUTC
        for i in range(1, len(x)):
            if x[i] < x[i - 1]:
                x[i] = x[i - 1]
        h1_area = (x[-1] - x[0]) * h1_avg_acc
        autc = auc(x, y) - h1_area
        autcs.append(autc)

        if model_name == 'no hist':
            no_hist_autc = autc

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

    color_idx = 0
    for i in sorted_idxs:
        x = xs[i]
        y = ys[i]
        autc_improv = autc_improvs[i]
        if autc_improv >= 0:
            sign = '+'
        else:
            sign = ''
        model_name = model_names[i]
        model_names_sorted.append(model_name)
        model = models[model_name]
        if model_name == 'no hist':
            color = 'black'
        else:
            color = cmap(color_idx / len(model_names))
            color_idx += 1

        sample_weight = model['sample_weight']
        if model_name == 'sim_ann':
            sample_weight = ['%.3f' % i for i in sample_weight]
        cell_text += [sample_weight + ['%s%.1f%%' % (sign, autc_improv)]]

        ax.plot(x, y, marker='.', label=model_name, color=color)
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
        title = 'average tradeoffs, dataset=%s user_type=%s' % (dataset, user_type)
        ax.set_title(title)
        plt.savefig('%s\\plots.png' % log_dir, bbox_inches='tight')
    else:
        len_h = df_results.loc[0]['len']
        title = 'dataset=%s user=%s len(h)=%d' % (dataset, user_name, len_h)
        ax.set_title(title)
        plt.savefig('%s\\%s.png' % (log_dir, user_name), bbox_inches='tight')
    if show_tradeoff_plots:
        plt.show()
    plt.clf()


# dataset = 'assistment'
# version = '100 seeds'
# user_type = 'user_id'

dataset = 'salaries'
version = '1'
user_type = 'relationship'

individual_users = True
bin_size = 3

results_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\simulated annealing'
log_dir = '%s\\%s\\%s\\%s' % (results_dir, dataset, version, user_type)
models = get_model_dict()

if not individual_users:
    plot_results(log_dir, dataset, user_type, models, bin_size=bin_size)
else:
    users_dir = '%s\\%s\\%s\\%s\\user_logs' % (results_dir, dataset, version, user_type)
    if not os.path.exists(users_dir):
        safe_make_dir(users_dir)
        split_users(log_dir)
    user_ids = pd.unique(pd.read_csv('%s\\log.csv' % log_dir)['user'])
    for user_id in user_ids:
        plot_results('%s\\user_logs' % log_dir, dataset, user_type, models, bin_size=bin_size, user_name=user_id)

