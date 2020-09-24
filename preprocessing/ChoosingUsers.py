import pandas as pd
import numpy as np
import csv


def choose_users():
    dataset_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/%s' % dataset
    df = pd.read_csv('%s/%s.csv' % (dataset_dir, dataset))
    user_lens = df[user_col].value_counts()
    groups_by_user = df.groupby(user_col)
    with open('%s/%s chosen_users.csv' % (dataset_dir, dataset), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(df.columns)
        user_idx = 0

        if choose_first_n:
            for user, hist_len in user_lens.items():
                for i, row in groups_by_user.get_group(user).iterrows():
                    writer.writerow(row)
                user_idx += 1
                if user_idx == num_users:
                    break
        else:
            chosen_lens = np.array([i / (num_users - 1) for i in reversed(range(num_users))])
            chosen_lens = (chosen_lens * (max_user_len - min_user_len) + min_user_len).tolist()
            for user, hist_len in user_lens.items():
                if hist_len <= chosen_lens[user_idx]:
                    for i, row in groups_by_user.get_group(user).iterrows():
                        writer.writerow(row)
                    user_idx += 1
                    if user_idx == len(chosen_lens):
                        break


dataset = 'GZ'
user_col = 'user'

num_users = 300
choose_first_n = False

max_user_len = 20
min_user_len = 5


choose_users()
print('\ndone')



