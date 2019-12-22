import csv
import os.path
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import auc
# import tensorflow as tf
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import L1L2
import seaborn as sn

import Models


def make_monotonic(x, y):
    i = 0
    while i < len(x)-1:
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


# Data-set paths

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\creditRiskAssessment.csv"
# target_col = 'RiskPerformance'
# dataset_fraction = 0.5
# threshold = 75
# users = {'1': df[:100].loc[df['ExternalRiskEstimate'] > threshold]}.items()

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\hospitalMortalityPrediction\\ADMISSIONS_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\hospitalMortalityPrediction.csv"
# target_col = 'HOSPITAL_EXPIRE_FLAG'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\recividismPrediction\\compas-scores-two-years_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\recividismPrediction.csv"
# target_col = 'is_recid'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\fraudDetection\\transactions.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\fraudDetection.csv"
# target_col = 'isFraud'

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\e-learning"
# target_col = 'correct'
# categ_cols = ['skill', 'tutor_mode', 'answer_type', 'type']
# user_group_names = ['user_id']
# skip_cols = []
# df_max_size = 100000
# layers = [100]

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

full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\salaries\\salaries.csv'
results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\salaries"
target_col = 'salary'
original_categ_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
skip_cols = []
df_max_size = -1
layers = []

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\abalone\\abalone.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\abalone"
# target_col = 'Rings'
# categ_cols = []
# user_group_names = ['sex']

# selecting experiment parameters

history_train_fraction = 0.5
h1_train_size = 200
h2_train_size = 5000
test_size = int(h2_train_size * history_train_fraction)

h1_epochs = 500
h2_epochs = 200

batch_size = 128
regularization = 0

seeds = range(5)
# seeds = [0]

diss_count = 10
normalize_diss_weight = True
if normalize_diss_weight:
    diss_weights = [(i + i / (diss_count - 1)) / diss_count for i in range(diss_count)]
    diss_multiply_factor = 1
else:
    diss_weights = range(diss_count)
    diss_multiply_factor = 1.0


range_stds = range(-30, 30, 2)
hybrid_stds = list((-x/10 for x in range_stds))

min_history_size = 100
max_history_size = 100000
current_user_count = 0
user_max_count = 30

# only_L1 = True
# only_hybrid = False
# # hybrid_method = 'stat'
hybrid_method = 'nn'

model_names = ['no hist', 'L3', 'hybrid']
colors = ['k', 'r', 'g']

split_by_chronological_order = False
copy_h1_weights = False
balance_histories = True
plot_confusion = False
get_area = True

only_train_h1 = False

show_plots = True

# if only_hybrid:
#     diss_weights = [0]

# if True:
for categ in original_categ_cols:
    if categ not in ['marital-status']:
        continue
    user_group_names = [categ]
    categ_cols = original_categ_cols.copy()
    categ_cols.remove(categ)

    plots_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots\\'+categ
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    else:
        os.makedirs(plots_dir)
    # os.makedirs(plots_dir + '\\by_hist_length')
    # os.makedirs(plots_dir + '\\by_accuracy_range')
    # os.makedirs(plots_dir + '\\by_compatibility_range')
    os.makedirs(plots_dir + '\\by_user_id')
    os.makedirs(plots_dir + '\\model_training')

    with open(plots_dir + '\\log.csv', 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        header = ['train frac', 'user_id', 'instances', 'train seed', 'comp range', 'acc range', 'h1 acc',
                  'diss weight']
        for name in model_names:
            if name != 'hybrid':
                header += [name+' x', name+' y']
        writer.writerow(header)

    if 'hybrid' in model_names:
        with open(plots_dir + '\\hybrid_log.csv', 'w', newline='') as file_out:
            writer = csv.writer(file_out)
            header = ['train frac', 'user_id', 'instances', 'train seed', 'std offset', 'hybrid x', 'hybrid y']
            writer.writerow(header)

    print('loading data...')
    df_full = pd.read_csv(full_dataset_path)
    if df_max_size >= 0:
        df_full = df_full[:df_max_size]

    for col in skip_cols:
        try:
            del df_full[col]
        except:
            pass

    # one hot encoding
    print('pre-processing data... ')
    ohe = ce.OneHotEncoder(cols=categ_cols, use_cat_names=True)
    df_full = ohe.fit_transform(df_full)
    # df_balanced = ohe.transform(df_balanced)
    print('num columns = %d'%df_full.shape[1])

    print('splitting into train and test sets...')

    # create user groups
    user_groups_train = []
    user_groups_test = []
    for user_group_name in user_group_names:
        user_groups_test += [df_full.groupby([user_group_name])]

    # separate histories into training and test sets
    students_group = user_groups_test[0]
    if split_by_chronological_order:
        df_train = students_group.apply(lambda x: x[:int(len(x) * history_train_fraction) + 1])
    else:
        df_train = students_group.apply(lambda x: x.sample(n=int(len(x) * history_train_fraction) + 1, random_state=1))

    df_train.index = df_train.index.droplevel(0)
    df_test = df_full.drop(df_train.index)
    user_groups_test[0] = df_test.groupby([user_group_names[0]])
    user_groups_train += [df_train.groupby([user_group_names[0]])]
    del df_train[user_group_names[0]]
    del df_test[user_group_names[0]]

    del df_full

    # balance sets
    print('balancing train set...')

    target_group = df_train.groupby(target_col)
    df_train = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))
    df_train = df_train.reset_index(drop=True)

    if balance_histories:

        # target_group = df_train.groupby(target_col)
        # df_train = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))
        # df_train = df_train.reset_index(drop=True)

        target_group = df_test.groupby(target_col)
        df_test = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))
        df_test = df_test.reset_index(drop=True)

    df_train_subsets_by_seed = []
    Xs_by_seed = []
    Ys_by_seed = []
    h1s_by_seed = []
    h2s_not_using_history_by_seed = []

    tests_group = {}
    tests_group_user_ids = []

    if only_train_h1:
        print('\nusing cross validation\n')
        train_sizes = [200, 5000]
        # train_sizes = [i * 100 for i in range(2, 11)]
        h1_epochs = 600
        seeds = range(1)
        n_features = int(df_train.shape[1])-1
        layers = []
        # layers = [int(n_features/2)]
        # layers = [90, 40]
        regularization = 0
        bottom, top = 0.4, 1.02
    else:
        train_sizes = [h1_train_size]

    for h1_train_size in train_sizes:

        if only_train_h1:
            h2_train_size = h1_train_size + 200

        train_accuracies = pd.DataFrame()
        test_accuracies = pd.DataFrame()

        start_time = int(round(time.time() * 1000))

        for seed in seeds:
            print('---\nSETTING TRAIN SEED '+str(seed)+'...\n---\n')
            if not only_train_h1:
                df_train_subset = df_train.sample(n=h2_train_size, random_state=seed)
            else:
                df_train_subset = df_train.sample(n=h2_train_size, random_state=seed).reset_index(drop=True)
                # df_train_subset = df_train.sample(n=h2_train_size, random_state=0).reset_index(drop=True)
                test_size = h2_train_size - h1_train_size
                test_range = range(test_size)
                # test_range = range(seed * test_size, (seed + 1) * test_size)
                train_part = df_train_subset.drop(test_range)
                test_part = df_train_subset.loc[test_range]
                df_train_subset = train_part.append(test_part)

            df_train_subsets_by_seed += [df_train_subset]

            # tests_group[str(seed)] = df_train_subset
            tests_group[str(seed)] = df_test.sample(n=test_size, random_state=seed)
            tests_group_user_ids += [str(seed)]

            X = df_train_subset.loc[:, df_train_subset.columns != target_col]
            Y = df_train_subset[[target_col]]

            scaler = MinMaxScaler()
            X = scaler.fit_transform(X, Y)
            labelizer = LabelBinarizer()
            Y = labelizer.fit_transform(Y)

            Xs_by_seed += [X]
            Ys_by_seed += [Y]

            h1 = Models.NeuralNetwork(X, Y, h1_train_size, h1_epochs, batch_size, layers, 0.02,
                               weights_seed=1, plot_train=True, regularization=regularization)
            tf.reset_default_graph()
            h1s_by_seed += [h1]

            train_accuracies[seed] = h1.plot_train_accuracy
            test_accuracies[seed] = h1.plot_test_accuracy

            if not only_train_h1:
                print("training h2s not using history...")
                h2s_not_using_history = []
                first_diss = True
                for diss_weight in diss_weights:
                    print('dissonance weight ' + str(len(h2s_not_using_history) + 1) + "/" + str(len(diss_weights)))

                    # if not first_diss and only_L1:
                    #     h2s_not_using_history += [h2s_not_using_history[0]]
                    #     continue

                    h2s_not_using_history += [Models.NeuralNetwork(X, Y, h2_train_size, h2_epochs, batch_size, layers, 0.02,
                                                                   diss_weight, h1, 'D', make_h1_subset=False,
                                                                   test_model=False,
                                                                   copy_h1_weights=copy_h1_weights, weights_seed=2,
                                                                   normalize_diss_weight=normalize_diss_weight)]
                    tf.reset_default_graph()
                    first_diss = False
                h2s_not_using_history_by_seed += [h2s_not_using_history]

        if show_plots:
            plot_x = list(range(h1_epochs))
            plt.plot(plot_x, train_accuracies.mean(axis=1), label='train accuracy')
            plt.plot(plot_x, test_accuracies.mean(axis=1), label='test accuracy')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend()
            plt.grid()
            if only_train_h1:
                plt.ylim(bottom, top)
            runtime = int((round(time.time() * 1000)) - start_time) / 60000
            plt.title('seed='+str(seed)+' train=' + str(h1_train_size) + ' test=' + str(h2_train_size - h1_train_size) +
                      ' epochs=' + str(h1_epochs) + ' run=%.2f min' % runtime + '\nlayers=' + str(layers)
                      + ' reg=' + str(regularization))
            plt.savefig(plots_dir + '\\model_training\\' + 'h1_train_seed_' + str(seed))
            if show_plots:
                plt.show()
            plt.clf()

    if only_train_h1:
        exit()

    del df_train
    del df_test

    user_group_names.insert(0, 'test')
    user_groups_test.insert(0, tests_group)

    students_group_user_ids = list(students_group.groups.keys())
    user_groups_user_ids = [tests_group_user_ids, students_group_user_ids]
    user_group_idx = -1

    for user_group_test in user_groups_test:
        user_group_idx += 1
        user_group_name = user_group_names[user_group_idx]
        user_ids = user_groups_user_ids[user_group_idx]

        total_users = 0
        user_ids_in_range = []

        user_test_sets = {}
        user_train_sets = {}

        if user_group_name == 'test':
            for seed in seeds:
                total_users += 1
                user_ids_in_range += [str(seed)]
                user_test_sets[str(seed)] = user_group_test[str(seed)]
        else:
            for user_id in user_ids:
                try:
                    user_test_set = user_group_test.get_group(user_id)
                    user_train_set = user_groups_train[user_group_idx - 1].get_group(user_id)
                except KeyError:
                    continue

                if balance_histories:
                    target_group = user_test_set.groupby(target_col)
                    if len(target_group.size()) == 1:
                        continue
                    user_test_set = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))

                    target_group = user_train_set.groupby(target_col)
                    if len(target_group.size()) == 1:
                        continue
                    user_train_set = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))

                history_len = len(user_test_set) + len(user_train_set)
                if min_history_size <= history_len <= max_history_size:
                    total_users += 1
                    user_ids_in_range += [user_id]
                    user_test_sets[user_id] = user_test_set
                    user_train_sets[user_id] = user_train_set

        user_count = 0
        # for user_id, user_test_set in user_group_test:
        for user_id in user_ids_in_range:
            user_test_set = user_test_sets[user_id]
            if not user_group_name == 'test' and user_count == user_max_count:
                break

            user_count += 1
            if user_count <= current_user_count:
                continue

            history_test_x = scaler.transform(user_test_set.loc[:, user_test_set.columns != target_col])
            history_test_y = labelizer.transform(user_test_set[[target_col]])

            if user_group_name != 'test':
                user_train_set = user_train_sets[user_id]
                history_len = len(user_test_set) + len(user_train_set)

                history_train_x = scaler.transform(user_train_set.loc[:, user_train_set.columns != target_col])
                history_train_y = labelizer.transform(user_train_set[[target_col]])

                history = Models.History(history_train_x, history_train_y, 0.001)

            else:
                history_len = len(user_test_set)

            for seed_idx in range(len(seeds)):
                seed = seeds[seed_idx]
                if user_group_name == 'test' and seed_idx != user_count-1:
                    continue
                print(str(user_count) + '/' + str(total_users) + ' ' + user_group_name + ' ' + str(user_id) +
                      ', instances: ' + str(history_len) + ', seed='+str(seed)+'\n')

                confusion_dir = plots_dir + '\\confusion_matrixes\\'+user_group_name+'_'+str(user_id)+'\\seed_'+str(seed)
                if plot_confusion:
                    if not os.path.exists(confusion_dir):
                        os.makedirs(confusion_dir)

                X = Xs_by_seed[seed_idx]
                Y = Ys_by_seed[seed_idx]
                h1 = h1s_by_seed[seed_idx]
                h2s_not_using_history = h2s_not_using_history_by_seed[seed_idx]

                # test h1 on user
                result = h1.test(history_test_x, history_test_y)
                h1_acc = result['auc']

                if plot_confusion:
                    title = user_group_name +'=' + str(user_id) +' h1 y='+'%.2f' % (h1_acc)
                    path = confusion_dir + '\\h1_seed_'+str(seed)+'.png'
                    plot_confusion_matrix(result['predicted'], history_test_y, title, path)

                if user_group_name != 'test':

                    if 'L0' in model_names:
                        history.set_simple_likelihood(X, magnitude_multiplier=2)
                        # history.set_simple_likelihood(X, h2s_not_using_history[0].W1, magnitude_multiplier=2)
                    if {'L1', 'L2'}.intersection(set(model_names)):
                        history.set_kernels(X, magnitude_multiplier=10)

                    models_x = []
                    models_y = []

                    for i in range(len(model_names)):
                        model_name = model_names[i]
                        print('model ' + str(i + 1) + "/" + str(len(model_names))+': '+model_name+'\n')
                        model_x = []
                        model_y = []
                        models_x += [model_x]
                        models_y += [model_y]
                        weights = diss_weights

                        if model_name == 'hybrid':
                            weights = hybrid_stds
                            h2s_not_using_history[0].set_hybrid_test(history, history_test_x, hybrid_method, layers)

                        for j in range(len(weights)):
                            if model_name == 'no hist':
                                result = h2s_not_using_history[j].test(history_test_x, history_test_y, h1)
                            else:
                                weight = weights[j]
                                if model_name == 'hybrid':
                                    result = h2s_not_using_history[0].hybrid_test(history_test_y, weight)
                                else:
                                    print('weight ' + str(j + 1) + "/" + str(len(weights)))
                                    tf.reset_default_graph()
                                    h2 = Models.NeuralNetwork(X, Y, h2_train_size, h2_epochs, batch_size, layers, 0.02, weight, h1, 'D',
                                                              history=history, use_history=True, model_type=model_name, test_model=False,
                                                              copy_h1_weights=copy_h1_weights, weights_seed=2,
                                                              normalize_diss_weight=normalize_diss_weight)
                                    result = h2.test(history_test_x, history_test_y, h1)

                            model_x += [result['compatibility']]
                            model_y += [result['auc']]

                            if plot_confusion:
                                title = user_group_name + '=' + str(user_id) + ' model='+model_name \
                                        +' x='+'%.2f' % (result['compatibility']) +' y='+'%.2f' % (result['auc'])
                                path = confusion_dir + '\\'+model_name+'_seed_'+str(seed)+'_' + str(j) + '.png'
                                plot_confusion_matrix(result['predicted'], history_test_y, title, path)

                    min_x = min(min(i) for i in models_x)
                    min_y = min(min(i) for i in models_y)
                    max_x = max(max(i) for i in models_x)
                    max_y = max(max(i) for i in models_y)

                    if get_area:
                        mono_xs = [i.copy() for i in models_x]
                        mono_ys = [i.copy() for i in models_y]

                        for i in range(len(mono_xs)):
                            make_monotonic(mono_xs[i], mono_ys[i])

                        h1_area = (1-min_x)*h1_acc
                        
                        areas = [auc([min_x] + mono_xs[i] + [1], [mono_ys[i][0]] + mono_ys[i] + [h1_acc]) - h1_area
                                 for i in range(len(mono_xs))]
                        
                    com_range = max_x - min_x
                    auc_range = max_y - min_y

                    # plotting
                    h1_x = [min_x, max_x]
                    h1_y = [h1_acc, h1_acc]
                    plt.plot(h1_x, h1_y, 'k--', marker='.', label='h1')

                    for i in range(len(model_names)):
                        plt.plot(models_x[i], models_y[i], colors[i], marker='.', label=model_names[i])

                    plt.xlabel('compatibility')
                    plt.ylabel('accuracy')
                    plt.grid()
                    plt.legend()
                    title = 'user=' + str(user_id) + ' hist_len=' + str(history_len) + ' split=' \
                            + str(history_train_fraction) + ' seed=' + str(seed)
                    plt.title(title)
                    plt.savefig(plots_dir+'\\by_user_id\\'+user_group_name+'_'+str(user_id)+'_seed_'+str(seed)+'.png')

                    if plot_confusion:
                        plt.savefig(confusion_dir + '\\plot.png')
                    if show_plots:
                        plt.show()

                    # write log
                    hybrid_idx = model_names.index('hybrid')
                    with open(plots_dir + '\\log.csv', 'a', newline='') as file_out:
                        writer = csv.writer(file_out)
                        for i in range(len(diss_weights)):
                            row = [str(history_train_fraction), str(user_id), str(history_len), str(seed),
                                   str(com_range), str(auc_range), str(h1_acc), str(diss_weights[i])]
                            for j in range(len(model_names)):
                                if j == hybrid_idx:
                                    continue
                                row += [models_x[j][i]]
                                row += [models_y[j][i]]
                            writer.writerow(row)

                    with open(plots_dir + '\\hybrid_log.csv', 'a', newline='') as file_out:
                        writer = csv.writer(file_out)
                        for i in range(len(hybrid_stds)):
                            row = [str(history_train_fraction), str(user_id), str(history_len), str(seed),
                                   str(hybrid_stds[i]), str(models_x[hybrid_idx][i]), str(models_y[hybrid_idx][i])]
                            writer.writerow(row)

                else:  # on test
                    h2_x = []
                    h2_y = []
                    for i in range(len(h2s_not_using_history)):
                        result = h2s_not_using_history[i].test(history_test_x, history_test_y, h1)
                        h2_x += [result['compatibility']]
                        h2_y += [result['auc']]

                        if plot_confusion:
                            title = user_group_name + '=' + str(user_id) + \
                                    ' h2=no_hist x='+'%.2f' % (result['compatibility']) +' y='+'%.2f' % (result['auc'])
                            path = confusion_dir + '\\test_seed_'+str(seed)+'_'+str(i)+'.png'
                            plot_confusion_matrix(result['predicted'], history_test_y, title, path)

                    h1_x = [min(h2_x), max(h2_x)]
                    h1_y = [h1_acc, h1_acc]
                    plt.plot(h1_x, h1_y, 'k--', marker='.', label='h1')
                    plt.plot(h2_x, h2_y, 'b', marker='.', label='h2')
                    plt.xlabel('compatibility')
                    plt.ylabel('accuracy')
                    plt.legend()
                    # plt.legend(loc='lower left')
                    plt.title('test seed=' + str(user_id) + ' h1=' + str(h1_train_size) + ' h2=' + str(
                        h2_train_size)+' len='+str(history_len))

                    plt.savefig(plots_dir+'\\test_for_seed_' + str(user_id) + '.png')
                    if plot_confusion:
                        plt.savefig(confusion_dir+'\\plot.png')
                    plt.show()
                plt.clf()
