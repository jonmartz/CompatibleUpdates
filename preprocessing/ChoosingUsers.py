import pandas as pd
import numpy as np
import csv
import os


def choose_users():
    dataset_dir = 'C:/Users/Jonma/Documents/BGU/Thesis/DataSets/%s' % dataset
    df = pd.read_csv('%s/%s all users.csv' % (dataset_dir, dataset))
    user_lens = df[user_col].value_counts()
    groups_by_user = df.groupby(user_col)
    total_len = 0
    with open('%s/%s.csv' % (dataset_dir, dataset), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(df.columns)
        user_idx = 0

        if choose_first_n:
            for user, hist_len in user_lens.items():
                print('len = %d' % hist_len)
                for i, row in groups_by_user.get_group(user).iterrows():
                    writer.writerow(row)
                user_idx += 1
                if user_idx == num_users:
                    break
        else:
            if num_users > 0:
                chosen_lens = np.array([i / (num_users - 1) for i in reversed(range(num_users))])
                chosen_lens = (chosen_lens * (max_user_len - min_user_len) + min_user_len).tolist()
            for user, hist_len in user_lens.items():
                if num_users > 0:
                    if hist_len <= chosen_lens[user_idx]:
                        print('len = %d' % hist_len)
                        total_len += hist_len
                        for i, row in groups_by_user.get_group(user).iterrows():
                            writer.writerow(row)
                        if num_users > 0:
                            user_idx += 1
                            if user_idx == len(chosen_lens):
                                break
                elif min_user_len <= hist_len <= max_user_len:
                    print('len = %d' % hist_len)
                    total_len += hist_len
                    for i, row in groups_by_user.get_group(user).iterrows():
                        writer.writerow(row)
    print('\ntotal len = %d' % total_len)


def choose_timestamps():
    dataset_dir = 'C:/Users/Jonma/Documents/BGU/Thesis/DataSets/%s' % dataset
    df = pd.read_csv('%s/%s all users.csv' % (dataset_dir, dataset))
    dataset_dir += '/timestamp analysis'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if df_max_size > 0:
        df = df[:df_max_size]
    user_groups = df.groupby(user_col)
    user_hists = {user: hist for user, hist in user_groups if len(hist) > min_user_len}
    user_time_ranges = {user: [hist['timestamp'].min(), hist['timestamp'].max()]
                             for user, hist in user_hists.items()}
    users = list(user_hists.keys())
    if timestamp_from == -1:
        timestamp_min = df['timestamp'].min()
        timestamp_max = df['timestamp'].max()
    else:
        timestamp_min = timestamp_from
        timestamp_max = timestamp_to
        for user in users:
            time_range = user_time_ranges[user]
            if time_range[0] > timestamp_max or time_range[1] < timestamp_min:
                del user_hists[user], user_time_ranges[user]
    timestamps = np.linspace(timestamp_min, timestamp_max, n_splits)
    if get_only_constant_users:
        constant_users = {user: {'train': [], 'test': []} for user in user_hists.keys()}
    with open('%s/timestamp_splits.csv' % dataset_dir, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'n_users', 'train_len', 'test_len'])
        for i, timestamp in enumerate(timestamps):
            if get_only_constant_users:
                timestamp_users = set()
            timestamp = int(timestamp)
            n_users, total_train_len, total_test_len = 0, 0, 0
            for user, hist in user_hists.items():
                time_range = user_time_ranges[user]
                if time_range[0] > timestamp or time_range[1] < timestamp:
                    continue
                train_len = ((hist['timestamp'] < timestamp) & (hist['timestamp'] >= timestamps[0] - max_horizon)).sum()
                if train_len < min_set_len:
                    continue
                test_len = ((hist['timestamp'] >= timestamp) & (hist['timestamp'] < timestamps[-1] + max_horizon)).sum()
                if test_len < min_set_len:
                    continue
                n_users += 1
                total_train_len += train_len
                total_test_len += test_len
                if get_only_constant_users:
                    timestamp_users.add(user)
                    user_data = constant_users[user]
                    user_data['train'].append(train_len)
                    user_data['test'].append(test_len)
            if get_only_constant_users:
                users = list(user_hists.keys())
                for user in users:
                    if user not in timestamp_users:
                        del user_hists[user], constant_users[user]
                print('\tconstant_users = %d' % len(constant_users))
            writer.writerow([timestamp, n_users, total_train_len, total_test_len])
            print('%d/%d timestamp=%d users=%d train_len=%d test_len=%d' % (
                i + 1, len(timestamps), timestamp, n_users, total_train_len, total_test_len))

    if get_only_constant_users:
        if min_num_changes > 0:
            total_samples = 0
            users = list(constant_users.keys())
            for user in users:
                x = constant_users[user]['train']
                num_changes = 0
                for i in range(len(x) - 1):
                    if x[i] != x[i + 1]:
                        num_changes += 1
                if num_changes < min_num_changes:
                    del constant_users[user]
                else:
                    train_len = constant_users[user]['train'][0]
                    test_len = constant_users[user]['test'][0]
                    total_samples += train_len + test_len
            print('\nusers_that_changed_%d_times=%d total_samples=%d' % (
                min_num_changes, len(constant_users), total_samples))
        with open('%s/constant_users.csv' % dataset_dir, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['user']
            for i in range(len(timestamps)):
                header.extend(['train %d' % i, 'test %d' % i])
            writer.writerow(header)
            for user, data in constant_users.items():
                row = [user]
                for i in range(len(timestamps)):
                    row.extend([data['train'][i], data['test'][i]])
                writer.writerow(row)
        if save_users:
            filter_array = df[user_col].isin(constant_users)
            filter_array = filter_array & (df['timestamp'] >= timestamps[0] - max_horizon)
            filter_array = filter_array & (df['timestamp'] < timestamps[-1] + max_horizon)
            df[filter_array].to_csv('%s/%s.csv' % (dataset_dir, dataset), index=False)



# dataset = 'assistment'
dataset = 'citizen_science'

user_col = 'user_id'
timestamp_split = True

max_user_len = 1000
min_user_len = 50

if timestamp_split:
    df_max_size = 0
    n_splits = 10
    min_set_len = 20
    timestamp_from = 50_000
    timestamp_to = 150_000
    max_horizon = 50_000
    get_only_constant_users = True
    min_num_changes = 5
    save_users = True
    choose_timestamps()
else:
    num_users = 50  # -1 to get all users depending on criterion
    choose_first_n = False
    choose_users()

print('\ndone')



