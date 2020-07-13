import pandas as pd
import csv


def remove_duplicates(dir, name):
    path_in = '%s/%s.csv' % (dir, name)
    df = pd.read_csv(path_in)
    with open(path_in, 'w', newline='') as file_out:
        writer = csv.writer(file_out)
        writer.writerow(list(df.columns))

        current_value = ''
        max_count = -1
        count = 0
        in_duplicate = False
        for i, row in df.iterrows():
            row_value = '%s %d %d' % (row['user'], row['seed'], row['inner_seed'])
            if current_value == '':
                current_value = row_value
            if row_value == current_value:
                count += 1
                if max_count != -1 and count > max_count:
                    if not in_duplicate:
                        in_duplicate = True
                        print('\tduplicate in row %d' % (i + 1))
            else:
                if max_count == -1:
                    max_count = count
                count = 1
                current_value = row_value
                in_duplicate = False
            if not in_duplicate:
                writer.writerow(row)


user_type = 'relationship'
for metric in ['acc', 'auc']:
    for subset in ['valid', 'test']:
        print('\nmetric=%s subset=%s' % (metric, subset))
        dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/lightsail/results/%s/%s' % (user_type, metric)
        name = '%s_log' % subset
        try:
            remove_duplicates(dir, name)
        except FileNotFoundError:
            print('\tmissing')

print('\ndone')
