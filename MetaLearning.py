import pandas as pd
import numpy as np
from Models import Model, evaluate_params
from sklearn.model_selection import KFold
from joblib import dump


def run_meta_learning(path, model_type, meta_ver, n_folds=5, normalize=True, tune_epochs=False, save_model=False):
    df = pd.read_csv(f'{path}/meta_dataset_ver_{meta_ver}.csv')
    df_original = df.copy()
    users = pd.unique(df['user'])

    if meta_ver == 1:
        labels = ['no hist', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
    else:
        labels = ['baseline_is_best', 'train_is_best', 'valid_is_best']
        best_train = pd.read_csv(f'{path}/best_models_train_bins.csv')['model']
        best_valid = pd.read_csv(f'{path}/best_models_valid_bins.csv')['model']

    if normalize:
        for col in df.columns:
            if col not in ['user'] + labels:
                df[col] = (df[col] - df[col].mean()) / df[col].std()  # mean normalization

    params = get_params(model_type, df, labels)
    test_scores = []
    rows = []
    best_params = None
    for user_idx, test_user in enumerate(users):
        df_test = df.loc[df['user'] == test_user].drop(columns='user')
        x_test, y_test = df_test.drop(columns=labels), df_test[labels]

        df_rest = df.loc[df['user'] != test_user]
        df_full_train = df_rest.drop(columns='user')
        x_full_train, y_full_train = df_full_train.drop(columns=labels), df_full_train[labels]
        rest_users = np.array([u for u in users if u != test_user])

        # if not (model_type == 'NN' and best_params):
        if not best_params:
            if params and ((model_type == 'NN' and tune_epochs) or isinstance(next(iter(params.values())), list)):
                folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=1).split(rest_users))
                cross_val_scores, cross_val_evaluated_params = [], []
                for fold_idx, fold in enumerate(folds):
                    indexes_train, indexes_valid = fold
                    users_train, users_valid = rest_users[indexes_train], rest_users[indexes_valid]
                    df_train = df_rest.loc[df['user'].isin(users_train)].drop(columns='user')
                    df_valid = df_rest.loc[df['user'].isin(users_valid)].drop(columns='user')
                    x_train, y_train = df_train.drop(columns=labels), df_train[labels]
                    x_valid, y_valid = df_valid.drop(columns=labels), df_valid[labels]
                    scores, evaluated_params = evaluate_params(model_type, x_train, y_train, x_valid, y_valid, 'acc',
                                                               params, tune_epochs=tune_epochs)
                    if model_type == 'NN':
                        mean_epochs = len(evaluated_params.epoch) - int(params['_fit_patience'])
                        train_acc = evaluated_params.history['acc'][mean_epochs - 1]
                        val_acc = evaluated_params.history['val_acc'][mean_epochs - 1]
                        cross_val_evaluated_params.append(mean_epochs)
                        cross_val_scores.append(val_acc)
                        print(f'\t\tbest_epochs:{mean_epochs} train_acc:{train_acc:.4f} valid_acc:{val_acc:.4f}')
                    else:
                        cross_val_evaluated_params.append(evaluated_params)
                        cross_val_scores.append([score.mean() for score in scores])
                if model_type == 'NN':
                    best_params = params.copy()
                    mean_epochs = np.mean(cross_val_evaluated_params)
                    mean_score = np.mean(cross_val_scores)
                    print(f'\tmean_epochs:{mean_epochs} mean_val_acc:{mean_score:.4f}')
                    for key in list(best_params.keys()):  # remove params for callback
                        if '_fit_' in key:
                            del best_params[key]
                    best_params['_fit_epochs'] = int(mean_epochs)
                    best_params['_fit_verbose'] = 0
                else:
                    mean_cross_val_scores = np.mean(cross_val_scores, axis=0)
                    best_params = evaluated_params[np.argmax(mean_cross_val_scores)]
            else:  # if any value is non-list, skip cross validation (except for NN which tunes only the num of epochs)
                best_params = params
            print(f'best_params:{best_params}')
        model = Model(model_type, params=best_params)
        model.fit(x_full_train, y_full_train)

        if save_model:
            if model_type == 'NN':
                model.predictor.save(f'{path}/meta_model nn.h5')
            else:
                dump(model, f'{path}/meta_model.joblib')
            return

        y_pred = model.predict_proba(x_test)
        pred_idx = int(np.argmax(y_pred))

        y_test = y_test.to_numpy().squeeze()
        # test_acc = np.array(y_pred.round() == y_test).mean()
        test_score = int(y_test[pred_idx] == 1)
        test_scores.append(test_score)

        # for i, label_probs in enumerate(y_pred):
        #     label = labels[int(np.argmax(label_probs))]
        label = labels[pred_idx]
        if meta_ver != 1:
            if label == 'baseline_is_best':
                label = 'no hist'
            elif label == 'train_is_best':
                label = best_train[len(rows)]
            else:  # label == 'valid_is_best'
                label = best_valid[len(rows)]
        # rows.append([test_user, i, label])
        rows.append([test_user, 0, label])

        print(f'user:{user_idx + 1}/{len(users)} label:{pred_idx}'
              f' score:{test_score} avg_score:{np.mean(test_scores):.4f}')

        # print(f'user:{user_idx + 1}/{len(users)} test_acc:{test_acc:.4f}'
        #       f' y_true:{y_test} pred_idx:{pred_idx} ')
    # print(f'\nmean score: {np.mean(test_scores):.4f}')
    pd.DataFrame(rows, columns=['user', 'seed', 'model']).to_csv(
        f'{path}/best_models_meta_ver_{meta_ver}.csv', index=False)

    df_original['meta_score'] = test_scores
    df_original.to_csv(f'{path}/meta_dataset_ver_{meta_ver} with meta_score.csv', index=False)




def get_best_from_meta_dataset(path, meta_ver):
    df_meta = pd.read_csv(f'{path}/meta_dataset_ver_{meta_ver}.csv')
    best_train = pd.read_csv(f'{path}/best_models_train_bins.csv')['model']
    best_valid = pd.read_csv(f'{path}/best_models_valid_bins.csv')['model']
    rows = []
    last_user, fold = '', 0
    for i, row in df_meta.iterrows():
        fold += 1
        if last_user != row['user']:
            fold = 0
            last_user = row['user']
        # using elif 'cause it can't be multi-label from now on
        if row['baseline_is_best'] == 1:
            best_model = 'no hist'
        elif row['train_is_best'] == 1:
            best_model = best_train[i]
        else:  # row['valid_is_best'] == 1:
            best_model = best_valid[i]
        rows.append([row['user'], fold, best_model])
    pd.DataFrame(rows, columns=['user', 'seed', 'model']).to_csv(
        f'{path}/best_models_meta-oracle_ver_{meta_ver}.csv', index=False)


def get_params(model_type, df, labels):
    params_by_model = {  # VALUES SHOULD ALL BE LISTS IF YOU WANT CROSS_VALIDATION
        'lr': {
            # 'solver': 'liblinear',
        },
        'ridge': {
            # 'solver': 'liblinear',
        },
        'svm': {
            'kernel': 'linear',
            # 'kernel': 'rbf',
            # 'kernel': 'sigmoid',
        },
        # 'tree': {'ccp_alpha': [0.0001, 0.0005, 0.001, 0.005]},
        'tree': {'ccp_alpha': list(np.concatenate([
            np.linspace(0.0001, 0.0009, 9),
            np.linspace(0.001, 0.009, 9),
            np.linspace(0.01, 0.09, 9),
            [0.1, 1.0]
        ]))},
        'randomforest': {
            # 'bootstrap': [True, False],
            # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            # 'max_features': ['auto'],
            # 'min_samples_leaf': [1, 2, 4],
            # 'min_samples_split': [2, 5, 10],
            'ccp_alpha': [0.0001, 0.001, 0.01, 0.1],
            'n_estimators': [10, 20],
            # 'n_estimators': list(np.linspace(1, 10, 9).astype(int)),
        },
        'NN': {
            'input_dim': df.shape[1] - len(labels) - 1,  # -1 is for user column
            'output_dim': len(labels),
            'layer_dims': [],
            'multilabel': True,
            '_fit_epochs': 135,
            '_fit_verbose': 0,
            # '_fit_min_delta': 0,
            # '_fit_patience': 50,
            # '_fit_monitor': 'val_loss',
        },
    }
    return params_by_model[model_type]


def where_is_meta_best():
    path = 'results where meta is best.csv'
    df = pd.read_csv(path)
    df = df.loc[df['meta is best'] == 1]
    models = ['no hist'] + [f'L{i}' for i in range(1, 9)]

    rows = []
    for i, row in df.iterrows():
        autcs = row[4:13]
        new_row = list(row[['user', 'len']])
        for agent in ['valid', 'meta', 'test']:
            agent_autc = row[f'best_{agent}']
            selected = [models[j] for j, autc in enumerate(autcs) if autc == agent_autc]
            new_row.append(', '.join(selected))
        rows.append(new_row)
    pd.DataFrame(rows, columns=['user', 'len', 'valid', 'meta', 'test']
                 ).sort_values('len', ascending=False).to_csv('users.csv', index=False)


if __name__ == '__main__':
    where_is_meta_best()
