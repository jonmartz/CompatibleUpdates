import numpy as np
from sklearn.metrics import auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier


class Model:
    def __init__(self, model_type, model_name='model', old_model=None, diss_weight=None, subset_weights=None,
                 hist_range=None, params=None, tune_epochs=False):
        self.model_type = model_type
        self.model_name = model_name
        self.old_model = old_model
        self.diss_weight = diss_weight
        self.subset_weights = subset_weights
        self.hist_range = hist_range
        self.params = params.copy() if params else {}
        self.tune_epochs = tune_epochs

        self.base_params, self.fit_params = {}, {}
        for key, value in list(self.params.items()):
            if '_base_' in key:
                del self.params[key]
                self.base_params[key[6:]] = value  # remove the '_base_' string
            elif '_fit_' in key:
                del self.params[key]
                self.fit_params[key[5:]] = value  # remove the '_fit_' string
        if model_type == 'dummy':
            # self.predictor = DummyClassifier(random_state=1, **self.params)
            self.predictor = DummyClassifier(**self.params)
        elif model_type == 'tree':
            self.predictor = DecisionTreeClassifier(random_state=1, **self.params)
        elif model_type == 'ridge':
            self.predictor = RidgeClassifierCV(**self.params)
        elif model_type == 'lr':
            self.predictor = LogisticRegression(random_state=1, **self.params)
        elif model_type == 'svm':
            self.predictor = SVC(random_state=1, **self.params)
        elif model_type == 'adaboost':
            self.predictor = AdaBoostClassifier(DecisionTreeClassifier(random_state=1, **self.base_params),
                                                random_state=1, **self.params)
        elif model_type == 'randomforest':
            self.predictor = RandomForestClassifier(random_state=1, **self.params)
        elif model_type == 'NN':
            self.predictor = get_NN(**self.params)
        else:
            raise ValueError('invalid model type')

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        if self.model_type == 'NN' and self.tune_epochs:
            validation_data = None
            fit_params = self.fit_params
            callbacks = None
            if valid_x is not None:
                import tensorflow as tf
                validation_data = (valid_x, valid_y)
                epochs = fit_params['epochs']
                del fit_params['epochs']  # avoid sending this param to callback
                callbacks = [tf.keras.callbacks.EarlyStopping(**self.fit_params)]
                fit_params = {'epochs': epochs, 'verbose': 0}
            return self.predictor.fit(train_x, train_y, sample_weight=self.get_sample_weights(train_x, train_y),
                                      validation_data=validation_data, callbacks=callbacks, **fit_params)
        else:
            sample_weights = self.get_sample_weights(train_x, train_y)
            self.predictor.fit(train_x, train_y, sample_weight=sample_weights, **self.fit_params)

    def get_sample_weights(self, x, y):
        if self.old_model is None:
            return None

        diss_weight = self.diss_weight
        general_loss, general_diss, hist_loss, hist_diss = self.subset_weights

        # getting old predictions
        old_predicted = np.round(self.old_model.predict(x))
        old_correct = np.equal(old_predicted, y).astype(int)

        if hist_loss == hist_diss == 0:  # no hist
            gen = (1 - diss_weight) * general_loss
            diss = diss_weight * old_correct * general_diss
        else:
            gen = (1 - diss_weight) * (general_loss + hist_loss * self.hist_range)
            diss = diss_weight * old_correct * (general_diss + hist_diss * self.hist_range)
            if diss.sum() == 0:  # h1 is incorrect on all history samples
                diss = diss_weight * (general_diss + hist_diss * self.hist_range)
        sample_weight = gen + diss
        return sample_weight

    def predict(self, x):
        if self.model_type == 'NN':
            return self.predict_proba(x).round()
        else:
            return self.predictor.predict(x)

    def predict_proba(self, x):
        if self.model_type == 'NN':
            import tensorflow as tf
            y_pred = self.predictor(tf.convert_to_tensor(x)).numpy()
            if len(y_pred.shape) == 2:
                y_pred = y_pred.squeeze()
            return y_pred
        else:
            y_pred = self.predictor.predict_proba(x)
            return [i[0][1] for i in y_pred]

    def score(self, x, y, metric):
        new_predicted = self.predict(x)
        # new_predicted_prob = self.predict(x)
        # new_predicted = np.round(new_predicted_prob)
        new_correct = np.equal(new_predicted, y).astype(int)
        if metric == 'acc':
            performance = np.mean(new_correct)
        elif metric == 'auc':
            y_true = y
            y_pred = new_predicted.copy()
            # if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
            #     y_pred = y_pred[:, 1]
            #     y_pred = 1 - y_pred.reshape(-1)
            labels = np.unique(y)
            if len(labels) == 1:  # to avoid error when calculating auc
                y_true = y_true.copy()
                y_pred = y_pred.copy()
                y_true[-1] = 1 - y_true[-1]
                y_pred[-1] = 1 - y_pred[-1]
            performance = roc_auc_score(y_true, y_pred)
        if self.old_model is None:  # testing pre-update model
            return {'y': performance, 'predicted': new_predicted}
        old_correct = np.equal(self.old_model.predict(x), y).astype(int)
        sum_old_correct = np.sum(old_correct)

        if sum_old_correct == 0:
            compatibility = 1  # no errors can be new - dissonance can only be 0
        else:
            compatibility = np.sum(old_correct * new_correct) / np.sum(old_correct)
        return {'x': compatibility, 'y': performance, 'predicted': new_predicted}


def get_NN(input_dim, output_dim, layer_dims, multilabel=False, name='NN'):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.models import Model
    from keras.layers import Dense, Input
    import tensorflow as tf
    # tf.get_logger().setLevel('INFO')
    tf.random.set_seed(1)
    input = Input(shape=(input_dim,), name='input')
    layer = input
    for layer_dim in layer_dims:
        layer = Dense(layer_dim, activation='relu')(layer)
    if multilabel:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    output = Dense(output_dim, activation=activation)(layer)
    model = Model(input, output, name=name)
    model.compile(optimizer='adam', loss=loss, metrics=['acc'])
    return model


def evaluate_params(model_type, train_x, train_y, valid_x, valid_y, metric, candidate_params, subset_weights=None,
                    old_model=None, hist_range=None, weights_num=5, get_autc=True, verbose=False, tune_epochs=False):
    if model_type == 'NN' and tune_epochs:
        scores, evaluated_params = evaluate_params_step(
            model_type, train_x, train_y, valid_x, valid_y, metric, None, subset_weights, old_model,
            hist_range, weights_num, candidate_params, [], [], get_autc=get_autc, verbose=verbose,
            tune_epochs=tune_epochs)
        return [scores], evaluated_params
    else:
        scores, evaluated_params = [], []
        evaluate_params_step(
            model_type, train_x, train_y, valid_x, valid_y, metric, candidate_params.copy(), subset_weights, old_model,
            hist_range, weights_num, {}, scores, evaluated_params, get_autc=get_autc, verbose=verbose,
            tune_epochs=tune_epochs)
        return scores, evaluated_params


def evaluate_params_step(model_type, train_x, train_y, valid_x, valid_y, metric, candidate_params, subset_weights,
                         old_model, hist_range, weights_num, params, scores, evaluated_params_list,
                         step=1, get_autc=True, verbose=False, tune_epochs=False):
    if candidate_params:  # continue recursion
        param_name, param_values = next(iter(candidate_params.items()))
        del candidate_params[param_name]
        for param_value in param_values:
            new_params = params.copy()
            new_params[param_name] = param_value
            result = evaluate_params_step(
                model_type, train_x, train_y, valid_x, valid_y, metric, candidate_params.copy(), subset_weights,
                old_model, hist_range, weights_num, new_params, scores, evaluated_params_list, step + 1,
                get_autc=get_autc, verbose=verbose, tune_epochs=tune_epochs)
            if result:
                score, evaluated_params = result
                scores.append(score)
                evaluated_params_list.append(evaluated_params)
                if verbose:
                    prefix = '\t' * step
                    print('%sscore=%.5f params=%s' % (prefix, score, evaluated_params))
                if model_type == 'NN' and np.isclose(score, 1.0):
                    break
    else:  # final recursion step
        if subset_weights is None:  # pre-update model
            model = Model(model_type, params=params, tune_epochs=tune_epochs)
        else:  # baseline or personalized model
            if get_autc:  # todo: doesn't work with NNs
                coms, pers = [], []
                for weight in np.linspace(0, 1, weights_num):
                    model = Model(model_type, '', old_model, weight, subset_weights, hist_range, params,
                                  tune_epochs=tune_epochs)
                    model.fit(train_x, train_y, valid_x, valid_y)
                    result = model.score(valid_x, valid_y, metric)
                    coms.append(result['x'])
                    pers.append(result['y'])
                for i in range(1, len(coms)):  # make coms monotonically increasing for auc computation
                    if coms[i] < coms[i - 1]:
                        coms[i] = coms[i - 1]
                return auc([0] + coms, [pers[0]] + pers), params  # correct leftwards for fairness
            else:
                model = Model(model_type, '', old_model, 0, subset_weights, hist_range, params, tune_epochs=tune_epochs)
        fit_return = model.fit(train_x, train_y, valid_x, valid_y)
        if model_type == 'NN' and tune_epochs:
            params = fit_return  # will be the optimal num of epochs (to not under/over-fit)
            score = fit_return.history['val_acc'][-model.fit_params['patience']]
        else:
            score = model.score(valid_x, valid_y, metric)['y']
        return score, params
