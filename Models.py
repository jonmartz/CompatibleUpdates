import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score


class History:
    """
    Class that implements the user's history, calculating means and vars.
    """

    def __init__(self, x_train, y_train, width_factor=0.01, epsilon=0.0000001):
        self.x_train = x_train
        self.y_train = y_train
        # self.x_test = x_test
        # self.y_test = y_test
        self.epsilon = epsilon
        self.width_factor = width_factor
        self.parametric_magnitude_multiplier = 1
        self.kernel_magnitude_multiplier = 1
        self.means = None
        self.norm_const = None
        self.inverse_cov = None
        self.likelihood = None
        self.kernels = None
        self.range = None

    def set_constants_for_likelihood(self):
        self.means = np.mean(self.x_train, axis=0)
        # covariance = np.cov(self.instances.T)
        # det = np.linalg.det(covariance)
        # self.norm_const = 1.0/len(self.means)
        # # self.norm_const = 1.0/(math.pow((2*math.pi), float(len(self.means))/2)*math.pow(det, 1.0/2))
        # self.inverse_cov = np.linalg.inv(covariance)

    def set_simple_likelihood(self, X, weights=None, sigma=1, magnitude_multiplier=1):
        self.set_constants_for_likelihood()

        distances = []
        for x in X:
            distances += [np.linalg.norm(x - self.means)]
        distances = np.array(distances)
        likelihood = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * np.square(distances / sigma)) * magnitude_multiplier
        self.likelihood = np.reshape(likelihood, (len(X), 1))

    def set_cheat_likelihood(self, df, threshold, likely_val=1):
        """
        Works only with credit risk data-set
        """
        self.likelihood = []
        for index, row in df.iterrows():
            if row['ExternalRiskEstimate'] > threshold:
                self.likelihood += [likely_val]
            else:
                self.likelihood += [0]
        self.likelihood = np.reshape(self.likelihood, (len(df), 1))

    def set_kernels(self, df, sigma=1, magnitude_multiplier=1):
        self.kernel_magnitude_multiplier = magnitude_multiplier
        distances = []
        for instance in df:
            entry = []
            for hist_instance in self.x_train:
                entry += [np.linalg.norm(instance - hist_instance)]
            distances += [entry]
        distances = np.array(distances)
        self.kernels = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
            -1 / 2 * np.square(distances / sigma)) * magnitude_multiplier

    def set_range(self, hist_range, h2_train_len):
        self.range = np.zeros(h2_train_len)
        for i in range(hist_range[0], hist_range[1]):
            self.range[i] = 1


class DecisionTree:

    def __init__(self, x_train, y_train, model_name, ccp_alpha, max_depth=None,
                 old_model=None, diss_weight=None, hist=None):
        self.model_name = model_name
        self.ccp_alpha = ccp_alpha
        self.old_model = old_model
        self.diss_weight = diss_weight
        self.hist = hist

        self.tree = DecisionTreeRegressor(max_depth=max_depth, random_state=1, ccp_alpha=ccp_alpha)
        sample_weight = self.get_sample_weight(x_train, y_train)
        self.tree.fit(x_train, y_train, sample_weight=sample_weight)

        if 'adaboost' in model_name:  # test on train set to set the new model's ensemble weight
            for model in [old_model, self]:
                new_predicted = model.tree.predict(x_train).reshape(len(y_train), 1)
                new_incorrect = 1 - np.equal(new_predicted, y_train).astype(int)
                pred_error = np.mean(new_incorrect)
                model.ensemble_weight = np.log((1.0 - pred_error) / pred_error)

    def get_sample_weight(self, x, y):
        sample_weight = None
        if self.old_model is None:
            return sample_weight

        diss_weight = self.diss_weight
        model_name = self.model_name
        hist = self.hist

        # getting old predictions
        old_predicted = np.round(self.old_model.predict(x))
        old_correct = np.equal(old_predicted, y).astype(int).reshape(-1)

        if hist is None:  # no history

            if model_name == 'adaboost':
                old_incorrect = 1 - old_correct
                sample_weight = (1 - diss_weight) + diss_weight * old_incorrect
            else:  # Ece's method
                sample_weight = (1 - diss_weight) + diss_weight * old_correct

        else:  # use history

            if model_name == 'L0':  # get likelihood from mean and std
                sample_weight = (1 - diss_weight) + diss_weight * old_correct * hist.likelihood

            elif model_name == 'L2':  # get likelihood from average of kernels
                kernel_likelihood = np.sum(hist.kernels, axis=1) / len(hist.y_train)
                sample_weight = (1 - diss_weight) + diss_weight * old_correct * kernel_likelihood

            elif model_name in ['L1', 'L3', 'L4', 'no diss']:  # these need dissonance from history

                if model_name == 'no diss':  # consider history without dissonance
                    sample_weight = (1 - diss_weight) + diss_weight * hist.range

                elif model_name == 'L1':  # get likelihood from weighted average of kernels
                    kernel_likelihood = np.sum(hist.kernels * hist.range * old_correct) / np.sum(hist.kernels)
                    sample_weight = (1 - diss_weight) + diss_weight * kernel_likelihood

                elif model_name == 'L3':  # consider only dissonance of history train set
                    sample_weight = (1 - diss_weight) + diss_weight * hist.range * old_correct

                elif model_name == 'L4':  # consider dissonance of both the general and history train sets
                    sample_weight = (1 - diss_weight) + diss_weight * old_correct * (1 + hist.range)

                elif model_name == 'L5':  # consider diss and loss of both the general and history train sets
                    sample_weight = ((1 - diss_weight) + diss_weight * old_correct) * (1 + hist.range)

        return sample_weight * 10

    def predict(self, x):
        return self.tree.predict(x).reshape(len(x), 1)

    def test(self, x, y, metric, old_model=None):

        new_predicted_prob = self.predict(x)

        if 'adaboost' not in self.model_name:
            new_predicted = np.round(new_predicted_prob)
            new_correct = np.equal(new_predicted, y).astype(int)
            if metric == 'acc':
                accuracy = np.mean(new_correct)
            elif metric == 'auc':
                accuracy = roc_auc_score(y, new_predicted_prob)
            if old_model is None:  # testing pre-update model
                return {'y': accuracy, 'predicted': new_predicted}

        old_predicted_prob = old_model.predict(x)
        old_correct = np.equal(np.round(old_predicted_prob), y).astype(int)

        if 'adaboost' in self.model_name:
            # normalize output to the range [-0.5, 0.5]
            new_predicted_prob = old_model.ensemble_weight * (old_predicted_prob - 0.5) \
                            + self.ensemble_weight * (new_predicted_prob - 0.5)
            # translate sign into a prediction of either 0 or 1
            new_predicted = np.greater(new_predicted_prob, 0).astype(int)
            new_correct = np.equal(new_predicted, y).astype(int)
            if metric == 'acc':
                accuracy = np.mean(new_correct)
            elif metric == 'auc':
                accuracy = roc_auc_score(y, 1 + new_predicted_prob)

        compatibility = np.sum(old_correct * new_correct) / np.sum(old_correct)
        return {'x': compatibility, 'y': accuracy, 'predicted': new_predicted}

    # def set_hybrid_test(self, hist, x_test):
    #
    #     x_train = hist.x_train
    #     y_train = hist.y_train
    #     old_predicted = self.old_model.predict(x_train)
    #     old_correct = np.equal(np.round(old_predicted), y_train)
    #     new_predicted = self.predict(x_train)
    #     new_incorrect = np.not_equal(np.round(new_predicted), y_train)
    #
    #     y_diss = (old_correct * new_incorrect).astype(int)
    #     version_chooser = DecisionTree(x_train, y_diss, 'version chooser', self.ccp_alpha)
    #
    #     self.dissonant_likelihood = version_chooser.predict(x_test)
    #     self.dissonant_likelihood_mean = self.dissonant_likelihood.mean()
    #     self.dissonant_likelihood_std = self.dissonant_likelihood.std()
    #     self.hybrid_old_predicted = self.old_model.predict(x_test)
    #     self.hybrid_new_predicted = self.predict(x_test)

    # def hybrid_test(self, y, std_offset):
    #     # get accuracy and compatibility
    #     likelihood = self.dissonant_likelihood
    #     threshold = self.dissonant_likelihood_mean + self.dissonant_likelihood_std * std_offset
    #     hybrid_output = np.where(likelihood < threshold, self.hybrid_new_predicted, self.hybrid_old_predicted)
    #     predicted = np.round(hybrid_output)
    #     hybrid_correct = np.equal(predicted, y).astype(int)
    #     old_correct = np.equal(np.round(self.hybrid_old_predicted), y).astype(int)
    #     if metric == 'acc':
    #         accuracy = np.mean(hybrid_correct)
    #     elif metric == 'auc':
    #         accuracy = roc_auc_score(y, hybrid_output)
    #     compatibility = np.sum(old_correct * hybrid_correct) / np.sum(old_correct)
    #     return {'x': compatibility, 'y': accuracy, 'predicted': predicted}


class ParametrizedTree(DecisionTree):

    def __init__(self, x_train, y_train, ccp_alpha, sample_weight_params, max_depth=None,
                 old_model=None, diss_weight=None, hist=None):
        self.sample_weight_params = sample_weight_params
        model_name = '%.4f %.4f %.4f %.4f' % tuple(sample_weight_params)
        super().__init__(x_train, y_train, model_name, ccp_alpha, max_depth, old_model, diss_weight, hist)

    def get_sample_weight(self, x, y):
        diss_w = self.diss_weight
        general_loss, general_diss, hist_loss, hist_diss = self.sample_weight_params

        # getting old predictions
        old_predicted = np.round(self.old_model.predict(x))
        old_correct = np.equal(old_predicted, y).astype(int).reshape(-1)

        if hist_loss == hist_diss == 0:  # no hist
            sample_weight = (1 - diss_w) * general_loss + diss_w * old_correct * general_diss
        else:
            hist_range = self.hist.range
            if general_diss == hist_diss == 0:  # no diss
                sample_weight = (1 - diss_w) * general_loss + diss_w * hist_loss * hist_range
            else:
                sample_weight = (1 - diss_w) * (general_loss + hist_loss * hist_range) \
                                + diss_w * old_correct * (general_diss + hist_diss * hist_range)

        return sample_weight * 10
