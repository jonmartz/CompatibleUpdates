import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import time
import numpy as np


def analyze_histories(dataset_path, user_directory, user_col):

    if not os.path.exists(user_directory):
        os.makedirs(user_directory)

    df = pd.read_csv(dataset_path)
    # df = pd.read_csv(dataset_path)[:200000]

    students = df.groupby([user_col])

    for users in [students]:
        df = []
        user_count = 0
        for user_id, user_instances in users:
            user_count += 1
            df += [len(user_instances)]
        df = pd.DataFrame(df, columns=['len'])

        sub_df = df.loc[:]
        # sub_df = df.loc[df['len'] <= 200]

        fig, ax = plt.subplots()
        mu = df.mean()
        sigma = df.std()
        textstr = '\n'.join((
            r'$\mathrm{n}=%.0f$' % (user_count,),
            r'$\mu=%.2f$' % (mu,),
            r'$\sigma=%.2f$' % (sigma,),
            r'$\mathrm{max}=%.0f$' % (df.max(),)))
        sub_df.hist(bins=100)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                 verticalalignment='top', horizontalalignment='right', bbox=props)
        plt.title('user history length')
        plt.xlabel('history length')
        plt.ylabel('frequency')
        plt.savefig(user_directory + '\\'+user_col+'_histories.png')
        # plt.show()


def analyze_features(dataset_path, results_path, user_col, categ_cols):
    df = pd.read_csv(dataset_path)

    try:
        del df[user_col]
    except KeyError:
        pass

    for col in categ_cols:
    # for col in df.columns:
        print(col + ' [' + str(df[col].min()) + ', ' + str(df[col].max()) + ']')
        if col not in categ_cols:
            sub_df = df
            bins = 100
            # if col == 'ms_first_response':
            #     sub_df = df.loc[(df['ms_first_response'] <= 150000) & (df['ms_first_response'] > 0)]
            #     bins = 150
            #
            # elif col == 'attempt_count':
            #     sub_df = df.loc[(df['attempt_count'] <= 10)]
            #     bins = 20
            #
            # elif col == 'correct' or col == 'original' or col == 'first_action':
            #     bins = 8
            #
            # elif col == 'hint_count':
            #     bins = 20
            #
            # elif col == 'hint_total':
            #     sub_df = df.loc[(df['hint_total'] <= 10)]
            #     bins = 20
            #
            # elif col == 'overlap_time':
            #     sub_df = df.loc[(df['overlap_time'] <= 200000) & (df['overlap_time'] > 0)]
            #     bins = 150
            #
            # else:
            #     continue

            # add stats
            x = df[col]
            fig, ax = plt.subplots()
            mu = x.mean()
            # median = np.median(x)
            sigma = x.std()
            textstr = '\n'.join((
                r'$\mu=%.2f$' % (mu,),
                # r'$\mathrm{med}=%.2f$' % (median,),
                r'$\sigma=%.2f$' % (sigma,)))
            sub_df.hist(column=col, bins=bins)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                     verticalalignment='top', horizontalalignment='right', bbox=props)
            plt.xlabel('value')
            plt.ylabel('frequency')
            plt.title(col)

            plt.savefig(results_path + '\\' + col + '.png')

        else:
            value_counts = df[col].value_counts()
            value_counts.plot(kind='bar')
            plt.xlabel('category')
            plt.ylabel('frequency')
            plt.title(col)
            plt.subplots_adjust(bottom=0.3)

            plt.savefig(results_path + '\\' + col + '.png')
            # plt.show()


def compare_history_train_and_history_test_sets(continuous_col_num_bins=20):
    # full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\user_skills.csv'
    full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_full_encoded_with_skill.csv'
    save_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\distributions\\all features"
    csv_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\distributions\\all features\\log.csv"
    target_col = 'correct'
    categ_cols = ['correct', 'skill', 'original', 'tutor_mode', 'answer_type', 'type']
    # discrete_cols = ['opportunity', 'attempt_count']
    discrete_cols = ['attempt_count']
    user_group_names = ['user_id']
    train_frac = 0.5
    repetitions = [0]
    min_history_size = 200
    max_history_size = 300

    # df_full = pd.read_csv(full_dataset_path)[:10000]
    df_full = pd.read_csv(full_dataset_path)

    try:
        del df_full['school_id']
        del df_full['teacher_id']
        del df_full['student_class_id']
    except:
        pass

    print('pre-processing data and splitting into train and test sets...')

    # create user groups
    user_groups_train = []
    user_groups_test = []
    for user_group_name in user_group_names:
        user_groups_test += [df_full.groupby([user_group_name])]

    # separate histories into training and test sets
    students_group = user_groups_test[0]
    df_train = students_group.apply(lambda x: x[:int(len(x) * train_frac) + 1])
    df_train.index = df_train.index.droplevel(0)
    user_groups_test[0] = df_full.drop(df_train.index).groupby([user_group_names[0]])
    user_groups_train += [df_train.groupby([user_group_names[0]])]

    user_group_idx = -1

    del df_full['user_id']

    for user_group_test in user_groups_test:
        user_group_idx += 1
        user_group_name = user_group_names[user_group_idx]

        print(user_group_name)

        total_users = 0
        # i = 0
        for user_id, test in user_group_test:
            # i += 1
            # print(str(i) + ' ' + str(user_id) + ' size='+str(len(test)))
            if len(test) >= min_history_size:
                total_users += 1

        print(str(total_users) + ' users')

        user_count = 0
        with open(csv_path, 'w', newline='') as file_out:
            writer = csv.writer(file_out)
            header = ['user_id'] + list(df_full.columns)
            writer.writerow(header)

            for user_id, test in user_group_test:
                if len(test) < min_history_size:
                    continue
                user_count += 1
                print(str(user_count) + '/' + str(total_users) + ' ' + user_group_name + ' ' + str(
                    user_id) + ', test = ' + str(len(test)))
                row = [str(user_id)]
                train = user_groups_train[user_group_idx - 1].get_group(user_id)

                del test['user_id']
                del train['user_id']

                for col in df_full.columns:
                    train_col = train[col]
                    test_col = test[col]
                    if col not in categ_cols:
                        num_bins = continuous_col_num_bins
                        if col in discrete_cols:
                            # num_bins = len(np.unique(np.concatenate((train_col, test_col))))
                            bins = np.unique(np.concatenate((train_col, test_col)))
                        else:
                            min_val = min(min(train_col), min(test_col))
                            max_val = max(max(train_col), max(test_col))
                            bin_width = (max_val-min_val)/num_bins
                            bins = [min_val+i*bin_width for i in range(num_bins+1)]
                        train_col = pd.cut(train_col, bins, include_lowest=True)
                        test_col = pd.cut(test_col, bins, include_lowest=True)

                    # fig, ax = plt.subplots()
                    train_values = train_col.value_counts() / len(train)
                    test_values = test_col.value_counts() / len(test)
                    merged_df = pd.concat([train_values, test_values], axis=1).fillna(0)
                    merged_df.columns = ['train', 'test']

                    train_vector = merged_df['train']
                    test_vector = merged_df['test']
                    cos_sim = dot(train_vector, test_vector) / (norm(train_vector) * norm(test_vector))
                    cos_sim_str = '%.4f' % (cos_sim,)

                    merged_df.plot.bar()
                    # plt.bar(merged_df)
                    plt.legend(loc='upper right')
                    plt.xlabel(col + ' id')
                    plt.ylabel('fraction from set')
                    hist_len = len(train) + len(test)
                    title = user_group_name + '=' + str(user_id) + ' col=' + col + ' split=' + str(train_frac) + ' sim=' + cos_sim_str
                    plt.title(title)
                    if col in categ_cols:
                        plt.subplots_adjust(bottom=0.2)
                    elif col in discrete_cols:
                        plt.subplots_adjust(bottom=0.4)
                    else:
                        plt.subplots_adjust(bottom=0.5)

                    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    # plt.text(0.05, 0.95, 'cos sim = '+cos_sim_str, transform=ax.transAxes,
                    #          verticalalignment='top', horizontalalignment='left', bbox=props)
                    # plt.text(0.05, 0.95, 'cos sim = ' + cos_sim_str,
                    #          verticalalignment='top', horizontalalignment='left', bbox=props)

                    directory = save_path + '\\' + col
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    plt.savefig(directory + '\\sim_' + cos_sim_str + '_' + user_group_name + '_' + str(user_id) + '.png')
                    # plt.show()
                    n = 100
                    if user_count % n == n - 1:
                        plt.close('all')
                    # plt.clf()

                    row += [cos_sim]
                writer.writerow(row)


def history_split_correlation_all_student_pairs():
    full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\user_skills.csv'
    save_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\distributions\\all skills"
    categ_cols = ['skill']
    user_group_names = ['user_id']
    train_frac = 0.5
    min_history_size = 300
    # max_history_size = 300

    # df_full = pd.read_csv(full_dataset_path)[:10000]
    df_full = pd.read_csv(full_dataset_path)

    try:
        del df_full['school_id']
        del df_full['teacher_id']
        del df_full['student_class_id']
    except:
        pass

    print('pre-processing data and splitting into train and test sets...')

    # create user groups
    user_groups_train = []
    user_groups_test = []
    for user_group_name in user_group_names:
        user_groups_test += [df_full.groupby([user_group_name])]

    # separate histories into training and test sets
    students_group = user_groups_test[0]
    df_train = students_group.apply(lambda x: x[:int(len(x) * train_frac) + 1])
    df_train.index = df_train.index.droplevel(0)
    user_groups_test[0] = df_full.drop(df_train.index).groupby([user_group_names[0]])
    user_groups_train += [df_train.groupby([user_group_names[0]])]

    user_group_idx = -1

    for user_group_test in user_groups_test:
        user_group_idx += 1
        user_group_name = user_group_names[user_group_idx]

        # print(user_group_name)

        total_users = 0
        # i = 0
        user_ids = []
        for user_id, test in user_group_test:
            # i += 1
            # print(str(i) + ' ' + str(user_id) + ' size='+str(len(test)))
            if len(test) >= min_history_size:
                total_users += 1
                user_ids += [user_id]
        total_users = int((total_users*(total_users-1))/2)

        # print(str(total_users) + ' users')

        path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\' \
               'distributions\\all skills\\train_percent_50\\user pairs split correlation\\correlation.csv'
        with open(path, 'w', newline='') as file_out:
            writer = csv.writer(file_out)
            writer.writerow(['user i', 'user j', 'column', 'train sim', 'test sim'])
            user_count = 0
            rows = []
            for i in range(len(user_ids)-1):
                user_i = user_ids[i]
                train_i = user_groups_train[0].get_group(user_i)
                test_i = user_groups_test[0].get_group(user_i)
                for j in range(i+1, len(user_ids)):
                    user_j = user_ids[j]
                    train_j = user_groups_train[0].get_group(user_j)
                    test_j = user_groups_test[0].get_group(user_j)
                    user_count += 1

                    # for col in df.columns:
                    for col in categ_cols:
                        row = [user_i, user_j, col]
                        i_values = train_i[col].value_counts() / len(train_i)
                        j_values = train_j[col].value_counts() / len(train_j)
                        merged_df = pd.concat([i_values, j_values], axis=1).fillna(0)
                        merged_df.columns = ['i', 'j']
                        i_vector = merged_df['i']
                        j_vector = merged_df['j']
                        cos_sim_train = dot(i_vector, j_vector) / (norm(i_vector) * norm(j_vector))
                        row += ['%.5f' % (cos_sim_train,)]

                        i_values = test_i[col].value_counts() / len(test_i)
                        j_values = test_j[col].value_counts() / len(test_j)
                        merged_df = pd.concat([i_values, j_values], axis=1).fillna(0)
                        merged_df.columns = ['i', 'j']
                        i_vector = merged_df['i']
                        j_vector = merged_df['j']
                        cos_sim_test = dot(i_vector, j_vector) / (norm(i_vector) * norm(j_vector))
                        row += ['%.5f' % (cos_sim_test,)]

                        print(str(user_count) + '/' + str(total_users) + ' user ' + str(user_i) + ' with user ' + str(
                            user_j)+' train sim '+'%.5f' % (cos_sim_train,)+' test sim '+'%.5f' % (cos_sim_test,))
                        rows += [row]
            writer.writerows(rows)


# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\matific.csv'
# user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\analysis\\users'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\mallzee\\mallzee.csv'
# user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\mallzee\\analysis'
# categ_cols = ['userResponse', 'Currency', 'TypeOfClothing', 'Gender', 'InStock', 'Brand', 'Colour']
# user_col = 'userID'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\matific.csv'
# user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\analysis'
# categ_cols = ['is_finished', 'activity_context', 'grade_code']
# user_col = 'account_id'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\moviesKaggle.csv'
# user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\analysis\\users'
# # categ_cols = ['original_language', 'rating']
# categ_cols = ['genre']
# user_col = 'userId'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\moviesKaggle.csv'
# user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\moviesKaggle\\analysis\\users'
# # categ_cols = ['original_language', 'rating']

dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\salaries\\salaries.csv'
user_directory = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\salaries\\analysis'
categ_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

if not os.path.exists(user_directory):
    os.makedirs(user_directory)

# user_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
# for user_col in user_cols:
#     analyze_histories(dataset_path, user_directory, user_col)

analyze_features(dataset_path, user_directory, '', categ_cols)
