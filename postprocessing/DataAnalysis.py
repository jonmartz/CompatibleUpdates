import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import wasserstein_distance
# from scipy.spatial.distance import cosine


def get_feat_importance(x, y):
    tree = DecisionTreeRegressor(random_state=1)
    tree.fit(x, y)
    y_pred = np.round(tree.predict(x))
    accuracy = np.mean(np.equal(y_pred, y).astype(int))
    feat_importance = tree.feature_importances_
    return feat_importance, accuracy


def get_cache(log_dir, user_col):
    params_path = '/'.join(log_dir.split('/')[:-1]) + '/parameters.csv'
    row = pd.read_csv(params_path).iloc[0]
    cache = '%s skip_%s max_len_%d min_hist_%d max_hist_%d chrono_%s balance_%s' % (
        user_col, '_'.join(eval(row['skip_cols'])), row['dataset_max_size'], row['min_hist_len'], row['max_hist_len'],
        row['chrono_split'], row['balance_histories']
    )
    return cache


def make_data_analysis(log_dir, dataset, user_col, target_col):
    if os.path.exists('%s\\wasserstein_distances.csv' % log_dir):
        return

    cache = get_cache(log_dir, user_col)
    dataset_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\%s\\caches\\%s' % (dataset, cache)
    dataset_path = '%s\\0.csv' % dataset_dir
    df = pd.read_csv(dataset_path)

    cols = list(df.drop(columns=[user_col, target_col]).columns) + [target_col]
    df = df[[user_col] + cols]

    scaler = MinMaxScaler()
    df_x = df.drop(columns=[target_col, user_col])
    cols_no_target = df_x.columns
    y = df[target_col]
    x_norm = scaler.fit_transform(df_x, y)
    df_norm = pd.DataFrame(x_norm, columns=df_x.columns)
    df_norm[target_col] = y
    df_norm[user_col] = df[user_col]

    gen_feat_importance, gen_acc = get_feat_importance(df_norm.drop(columns=[target_col, user_col]), df_norm[target_col])
    print('gen acc = %.5f' % gen_acc)
    users = list(pd.unique(df_norm[user_col]))
    user_groups = df_norm.groupby(user_col)
    wasserstein_distances = [[] for i in cols]
    # cosine_distances = [[] for i in cols]
    feat_importances = [[gen_feat_importance[i]] for i in range(len(cols_no_target))]
    for user in users:
        df_user = user_groups.get_group(user)
        for i in range(len(cols)):
            col = cols[i]
            wasserstein_distances[i].append(wasserstein_distance(df_norm[col], df_user[col]))
            # cosine_distances[i].append(1 - cosine(df_norm[col], df_user[col]))
        user_feat_importance, user_acc = get_feat_importance(df_user.drop(columns=[target_col, user_col]),
                                                             df_user[target_col])
        print('\tuser %s acc = %.5f' % (user, user_acc))
        for i in range(len(cols_no_target)):
            feat_importances[i].append(user_feat_importance[i])

    # write distances
    df_dict = {'user': users}
    for i in range(len(cols)):
        df_dict[cols[i]] = wasserstein_distances[i]
    pd.DataFrame(df_dict).to_csv('%s\\wasserstein_distances.csv' % log_dir, index=False)

    # df_dict = {'user': users}
    # for i in range(len(cols)):
    #     df_dict[cols[i]] = cosine_distances[i]
    # pd.DataFrame(df_dict).to_csv('%s\\cosine_distances.csv' % dataset_dir, index=False)

    # write feature importances
    df_dict = {'user': ['general'] + users}
    for i in range(len(cols_no_target)):
        df_dict[cols_no_target[i]] = feat_importances[i]
    df_dict[target_col] = [1] * (len(users) + 1)
    pd.DataFrame(df_dict).to_csv('%s\\feature_importances.csv' % log_dir, index=False)


# log_dir = 'C:/Users/Jonathan/Documents/BGU/Research/Thesis/current result/user'
# dataset = 'ednet'
# user_col = 'user'
# target_col = 'correct_answer'
#
# make_data_analysis(log_dir, dataset, user_col, target_col)