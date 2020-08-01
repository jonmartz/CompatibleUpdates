import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


def get_model(model_type, x_train, y_train, model_name='None', subset_weights=None, old_model=None, diss_weight=None,
              hist_range=None, ccp_alpha=None, max_depth=None, ridge_alpha=None):
    if model_type == 'tree':
        return CompatibleRegressionTree(x_train, y_train, ccp_alpha, model_name, subset_weights, max_depth,
                                        old_model, diss_weight, hist_range)
    elif model_type == 'ridge':
        return CompatibleRidge(x_train, y_train, model_name, subset_weights, old_model, diss_weight, hist_range, ridge_alpha)


class Model:
    def __init__(self, model_name=None, old_model=None, diss_weight=None, subset_weights=None, hist_range=None):
        if model_name == 'None' and subset_weights is not None:
            self.model_name = '%.4f %.4f %.4f %.4f' % tuple(subset_weights)
        else:
            self.model_name = model_name
        self.subset_weights = subset_weights
        self.old_model = old_model
        self.diss_weight = diss_weight
        self.hist_range = hist_range
        self.predictor = None

    def get_sample_weights(self, x, y, scale=10):
        if self.old_model is None:
            return None

        diss_weight = self.diss_weight
        general_loss, general_diss, hist_loss, hist_diss = self.subset_weights

        # getting old predictions
        old_predicted = np.round(self.old_model.predict(x))
        old_correct = np.equal(old_predicted, y).astype(int).reshape(-1)

        if hist_loss == hist_diss == 0:  # no hist
            gen = (1 - diss_weight) * general_loss
            diss = diss_weight * old_correct * general_diss
        else:
            gen = (1 - diss_weight) * (general_loss + hist_loss * self.hist_range)
            diss = diss_weight * old_correct * (general_diss + hist_diss * self.hist_range)
        sample_weight = gen + diss
        return sample_weight * scale

    def predict(self, x):
        return self.predictor.predict(x).reshape(len(x), 1)

    def test(self, x, y, metric):
        new_predicted_prob = self.predict(x)
        new_predicted = np.round(new_predicted_prob)
        new_correct = np.equal(new_predicted, y).astype(int)
        if metric == 'acc':
            accuracy = np.mean(new_correct)
        elif metric == 'auc':
            accuracy = roc_auc_score(y, new_predicted_prob)
        if self.old_model is None:  # testing pre-update model
            return {'y': accuracy, 'predicted': new_predicted}
        old_predicted_prob = self.old_model.predictor.predict(x).reshape(len(x), 1)
        old_correct = np.equal(np.round(old_predicted_prob), y).astype(int)
        sum_old_correct = np.sum(old_correct)
        if sum_old_correct == 0:
            compatibility = 1  # no errors can be new - dissonance can only be 0
        else:
            compatibility = np.sum(old_correct * new_correct) / np.sum(old_correct)
        return {'x': compatibility, 'y': accuracy, 'predicted': new_predicted}


class CompatibleRegressionTree(Model):
    def __init__(self, x_train, y_train, ccp_alpha, model_name='None', subset_weights=None, max_depth=None,
                 old_model=None, diss_weight=None, hist_range=None):
        super().__init__(model_name, old_model, diss_weight, subset_weights, hist_range)
        self.ccp_alpha = ccp_alpha
        self.predictor = DecisionTreeRegressor(max_depth=max_depth, random_state=1, ccp_alpha=ccp_alpha)
        self.predictor.fit(x_train, y_train, sample_weight=self.get_sample_weights(x_train, y_train))


class CompatibleRidge(Model):
    def __init__(self, x_train, y_train, model_name='None', subset_weights=None,
                 old_model=None, diss_weight=None, hist_range=None, ridge_alpha=1.0):
        super().__init__(model_name, old_model, diss_weight, subset_weights, hist_range)
        self.predictor = Ridge(alpha=ridge_alpha, random_state=1)
        self.predictor.fit(x_train, y_train, sample_weight=self.get_sample_weights(x_train, y_train))
