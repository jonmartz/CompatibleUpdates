import pandas as pd
import numpy as np
import csv


def choose_users():
    chosen_lens = np.array([i / num_users for i in reversed(range(num_users + 1))])
    chosen_lens = (chosen_lens * (max_user_len - min_user_len) + min_user_len).tolist()

    # len_delta = int((max_user_len - min_user_len) / num_users)
    # chosen_lens = [max_user_len - i * len_delta for i in range(num_users)]

    dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/DataSets/%s' % dataset
    df = pd.read_csv('%s/%s.csv' % (dir, dataset))
    user_lens = df[user_col].value_counts()
    groups_by_user = df.groupby(user_col)
    with open('%s/%s_chosen_users.csv' % (dir, dataset), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(df.columns)
        chosen_user_idx = 0
        for user, hist_len in user_lens.items():
            if hist_len <= chosen_lens[chosen_user_idx]:
                for i, row in groups_by_user.get_group(user).iterrows():
                    writer.writerow(row)
                chosen_user_idx += 1
                if chosen_user_idx == len(chosen_lens):
                    break


dataset = 'assistment'
user_col = 'user_id'
num_users = 99
max_user_len = 1100
min_user_len = 50

choose_users()
print('\ndone')



