import csv
import os.path
import shutil
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import auc
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import L1L2
import seaborn as sn
import math


class NeuralNet:
    """
    Class implementing a Neural Network, capable of training using the log loss + dissonance
    to be able to produce compatible updates.
    """

    def __init__(self, X_train, Y_train, X_test, Y_test, batch_size, layers, model_name, min_train_epochs=100,
                 max_train_epochs=-1, diss_weight=None, old_model=None, copy_h1_weights=True, history=None,
                 initial_std=1, weights_seed=1, regularization=0.0, plot_train=True, ensemble_lr=1.0, early_stop=True,
                 acc_tier_height=0.01, max_same_tier_spree=100):

        self.model_name = model_name
        self.old_model = old_model
        self.final_weights = []
        self.final_biases = []
        self.dissonant_likelihood = None

        # if history is not None:
        #     try:
        #         kernels_train = history.kernels[self.train_indexes]
        #     except TypeError:
        #         pass
        #     try:
        #         likelihood_train = history.likelihood[self.train_indexes]
        #     except TypeError:
        #         pass

        # ------------ #
        # CREATE MODEL #
        # ------------ #

        features_count = len(X_train[0])
        train_len = len(X_train)
        labels_dim = 1

        # these placeholders serve as the input tensors
        x = tf.placeholder(tf.float32, [None, features_count], name='input')
        y = tf.placeholder(tf.float32, [None, labels_dim], name='labels')
        likelihood = tf.placeholder(tf.float32, [None, labels_dim], name='likelihood')

        # input tensor for the base model predictions
        y_old_probabilities = tf.placeholder(tf.float32, [None, labels_dim], name='old_probabilities')
        y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='old_corrects')

        # set initial weights
        initial_weights = []
        initial_biases = []

        if old_model is None or not copy_h1_weights:
            for i in range(len(layers)):
                if i == 0:
                    initial_weights += [tf.truncated_normal([features_count, layers[i]], mean=0,
                                                            stddev=initial_std / np.sqrt(features_count),
                                                            seed=weights_seed)]
                    initial_biases += [tf.truncated_normal([layers[i]], mean=0,
                                                           stddev=initial_std / np.sqrt(features_count),
                                                           seed=weights_seed)]
                else:
                    initial_weights += [tf.random_normal([layers[i - 1], layers[i]], mean=0, stddev=initial_std,
                                                         seed=weights_seed)]
                    initial_biases += [tf.random_normal([layers[i]], mean=0, stddev=initial_std, seed=weights_seed)]

            if len(layers) == 0:
                last_layer = features_count
            else:
                last_layer = layers[-1]
            initial_weights += [tf.random_normal([last_layer, labels_dim], mean=0, stddev=initial_std,
                                                 seed=weights_seed)]
            initial_biases += [tf.random_normal([labels_dim], mean=0, stddev=initial_std, seed=weights_seed)]
        else:
            for i in range(len(layers)):
                initial_weights += [tf.convert_to_tensor(old_model.final_weights[i])]
                initial_biases += [tf.convert_to_tensor(old_model.final_biases[i])]

        # build layers
        weights = []
        biases = []
        activations = []
        for i in range(len(layers)):
            weights += [tf.Variable(initial_weights[i], name='weights_' + str(i + 1))]
            biases += [tf.Variable(initial_biases[i], name='biases_' + str(i + 1))]
            if i == 0:
                activations += [tf.sigmoid((tf.matmul(x, weights[i]) + biases[i]), name='activations_' + str(i + 1))]
            else:
                activations += [tf.sigmoid((tf.matmul(activations[i - 1], weights[i]) + biases[i]),
                                           name='activations_' + str(i + 1))]

        weights += [tf.Variable(initial_weights[-1], name='weights_' + str(len(layers) + 1))]
        biases += [tf.Variable(initial_biases[-1], name='biases_' + str(len(layers) + 1))]

        if len(layers) == 0:
            logits = tf.matmul(x, weights[-1]) + biases[-1]
        else:
            logits = tf.matmul(activations[-1], weights[-1]) + biases[-1]
        output = tf.nn.sigmoid(logits, name='output')

        # for non parametric compatibility
        if model_name in ['L1', 'L2', 'L3', 'L4', 'baseline']:
            kernels = tf.placeholder(tf.float32, [None, len(history.y_train)], name='kernels')
            hist_x = tf.placeholder(tf.float32, [None, features_count], name='hist_input')
            hist_y = tf.placeholder(tf.float32, [None, labels_dim], name='hist_labels')
            hist_y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='hist_old_corrects')

            hist_activations = []
            for i in range(len(layers)):
                if i == 0:
                    hist_activations += [tf.sigmoid((tf.matmul(hist_x, weights[i]) + biases[i]),
                                                    name='hist_activations_' + str(i + 1))]
                else:
                    hist_activations += [tf.sigmoid((tf.matmul(hist_activations[i - 1], weights[i]) + biases[i]),
                                                    name='hist_activations_' + str(i + 1))]

            if len(layers) == 0:
                hist_logits = tf.matmul(hist_x, weights[-1]) + biases[-1]
            else:
                hist_logits = tf.matmul(hist_activations[-1], weights[-1]) + biases[-1]

        if early_stop or plot_train:
            # model evaluation tensors
            correct_prediction = tf.equal(tf.round(output), y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

        # loss computation
        log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

        # dissonance computation
        if old_model is None:  # for training pre-update version
            loss = log_loss

        else:  # for training post-update version

            Y_train_old_probabilities = old_model.predict_probabilities(X_train)
            Y_train_old_labels = np.round(Y_train_old_probabilities)
            Y_train_old_correct = np.equal(Y_train_old_labels, Y_train).astype(int)

            # to decrease the loss obtained from non-dissonant instances by 1-diss_weight
            loss_weights = (1 - diss_weight) * (1 - y_old_correct) + y_old_correct

            if history is None:
                if 'adaboost' in model_name:  # else, Ece's method is performed
                    Y_train_old_incorrect = 1 - Y_train_old_correct
                    pred_error = np.mean(Y_train_old_incorrect)
                    old_ensemble_weight = ensemble_lr * np.log((1.0 - pred_error) / pred_error)
                    old_model.ensemble_weight = old_ensemble_weight

                    if 'comp' in model_name:  # give larger weights to h1's correct predictions
                        y_important = y_old_correct
                    else:  # give larger weights to h1's mistakes
                        y_important = 1 - y_old_correct
                    # loss_weights = (1.0/train_len) * tf.exp(old_ensemble_weight * y_important * diss_weight * 10)
                    loss_weights = (1 - diss_weight) * (1 - y_important) + y_important

                loss = loss_weights * log_loss

            else:  # use history

                dissonance = y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

                if model_name == 'L0':  # get likelihood from mean and std
                    loss = (1 - diss_weight) * log_loss + diss_weight * tf.reduce_mean(dissonance * likelihood)

                elif model_name == 'L2':  # get likelihood from average of kernels
                    kernel_likelihood = tf.reduce_sum(kernels, axis=1) / len(history.y_train)
                    loss = (1 - diss_weight) * log_loss + diss_weight * kernel_likelihood * tf.reduce_mean(
                        dissonance)

                elif model_name in ['L1', 'L3', 'L4', 'baseline']:  # these need dissonance from history
                    hist_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=hist_y,
                        logits=hist_logits,
                        name='hist_dissonance')
                    hist_dissonance = tf.reshape(hist_y_old_correct * hist_loss, [-1])

                    if model_name == 'baseline':  # consider history without dissonance
                        loss = (1 - diss_weight) * log_loss + diss_weight * tf.reduce_mean(hist_loss)

                    elif model_name == 'L1':  # get likelihood from weighted average of kernels
                        kernel_likelihood = tf.reduce_sum(kernels * hist_dissonance) / tf.reduce_sum(kernels)
                        loss = (1 - diss_weight) * log_loss + diss_weight * tf.reduce_mean(kernel_likelihood)

                    elif model_name == 'L3':  # consider only dissonance of history train set
                        loss = (1 - diss_weight) * log_loss + diss_weight * tf.reduce_mean(hist_dissonance)

                    elif model_name == 'L4':  # consider dissonance of both the general and history train sets
                        loss = loss_weights * log_loss + diss_weight * tf.reduce_mean(hist_dissonance)

        if regularization == 0:
            loss = tf.reduce_mean(loss)
        else:
            regularizers = tf.nn.l2_loss(weights[0])
            for i in range(1, len(weights)):
                regularizers += tf.nn.l2_loss(weights[i])
            loss = tf.reduce_mean(loss + regularization * regularizers)

        # prepare training
        train_step = tf.train.AdamOptimizer().minimize(loss)
        init_op = tf.global_variables_initializer()

        # ----------- #
        # TRAIN MODEL #
        # ----------- #

        with tf.Session() as sess:
            sess.run(init_op)
            batches = int(len(X_train) / batch_size)

            if early_stop or plot_train:
                self.train_loss = []
                self.train_acc = []
                self.test_loss = []
                self.test_acc = []
                if early_stop:
                    self.best_epoch = 0
                    # best_acc_tier = 0
                    best_acc_tier = -1
                    same_tier_spree = 0

            epoch = 0
            while epoch != max_train_epochs:
                epoch += 1
                epoch_loss = 0
                for batch in range(batches + 1):
                    batch_start = batch * batch_size
                    if batch_start == X_train.shape[0]:
                        continue  # in case the len of train set is a multiple of batch size
                    batch_end = min((batch + 1) * batch_size, X_train.shape[0])
                    X_batch = X_train[batch_start:batch_end]
                    Y_batch = Y_train[batch_start:batch_end]

                    if old_model is None:  # without compatibility
                        epoch_loss += sess.run([train_step, loss], feed_dict={x: X_batch, y: Y_batch})[1]

                    else:  # with compatibility
                        Y_batch_old_probabilities = Y_train_old_probabilities[batch_start:batch_end]
                        Y_batch_old_correct = Y_train_old_correct[batch_start:batch_end]
                        if history is None:  # no personalization
                            epoch_loss += sess.run([train_step, loss],
                                                   feed_dict={x: X_batch, y: Y_batch,
                                                              y_old_probabilities: Y_batch_old_probabilities,
                                                              y_old_correct: Y_batch_old_correct})[1]
                        else:
                            hist_Y_old_probabilities = old_model.predict_probabilities(history.x_train)
                            hist_Y_old_labels = np.round(hist_Y_old_probabilities)
                            hist_Y_old_correct = np.equal(hist_Y_old_labels, history.y_train).astype(int)
                            if model_name in ['L1', 'L2']:
                                kernels_batch = history.kernels[batch_start:batch_end]
                                epoch_loss += sess.run([train_step, loss],
                                                       feed_dict={x: X_batch, y: Y_batch,
                                                                  y_old_probabilities: Y_batch_old_probabilities,
                                                                  y_old_correct: Y_batch_old_correct,
                                                                  hist_x: history.x_train,
                                                                  hist_y: history.y_train,
                                                                  hist_y_old_correct: hist_Y_old_correct,
                                                                  kernels: kernels_batch})[1]
                            elif model_name == 'L0':
                                likelihood_batch = history.likelihood[batch_start:batch_end]
                                epoch_loss += sess.run([train_step, loss],
                                                       feed_dict={x: X_batch, y: Y_batch,
                                                                  y_old_probabilities: Y_batch_old_probabilities,
                                                                  y_old_correct: Y_batch_old_correct,
                                                                  likelihood: likelihood_batch})[1]
                            elif model_name in ['L3', 'L4', 'baseline']:
                                epoch_loss += sess.run([train_step, loss],
                                                       feed_dict={x: X_batch, y: Y_batch,
                                                                  y_old_probabilities: Y_batch_old_probabilities,
                                                                  y_old_correct: Y_batch_old_correct,
                                                                  hist_x: history.x_train,
                                                                  hist_y: history.y_train,
                                                                  hist_y_old_correct: hist_Y_old_correct})[1]
                self.train_loss += [epoch_loss]

                if early_stop or plot_train:
                    # train_acc = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
                    # test_acc = sess.run(accuracy, feed_dict={x: X_test, y: Y_test})
                    # self.train_acc += [train_acc]
                    # self.test_acc += [test_acc]

                    # check early stop condition
                    if epoch > min_train_epochs and early_stop:
                        # acc = train_acc
                        # acc = test_acc  # little cheat to avoid overfitting
                        if best_acc_tier == -1:
                            best_acc_tier = epoch_loss + acc_tier_height
                        # if acc > best_acc_tier - acc_tier_height:
                        if epoch_loss < best_acc_tier + acc_tier_height:
                            self.best_epoch = epoch
                            self.final_weights = [i.eval() for i in weights]
                            self.final_biases = [i.eval() for i in biases]
                        # if acc > best_acc_tier + acc_tier_height:
                        if epoch_loss < best_acc_tier - acc_tier_height:
                            # best_acc_tier = acc
                            best_acc_tier = epoch_loss
                            same_tier_spree = 0
                        else:
                            same_tier_spree += 1
                            if same_tier_spree == max_same_tier_spree:
                                break  # end training

            # if test_model:
            #     acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_test, y: Y_test})
            #     # self.auc = sklearn.metrics.roc_auc_score(Y_test, out)
            #     self.accuracy = acc
            #     # print("h1 acc = %.4f" % self.accuracy)

            # save parameters
            self.final_epochs = epoch
            if not early_stop:
                for i in range(len(weights)):
                    self.final_weights += [weights[i].eval()]
                    self.final_biases += [biases[i].eval()]

            if 'adaboost' in model_name:  # test on train set to set the new model's ensemble weight
                Y_train_new_probabilities = self.predict_probabilities(X_train)
                Y_train_new_labels = np.round(Y_train_new_probabilities)
                Y_train_new_incorrect = 1 - np.equal(Y_train_new_labels, Y_train).astype(int)

                pred_error = np.mean(Y_train_new_incorrect)
                new_ensemble_weight = ensemble_lr * np.log((1.0 - pred_error) / pred_error)
                self.ensemble_weight = new_ensemble_weight

        tf.reset_default_graph()

    def predict_probabilities(self, x):
        """
        predict the labels for dataset x
        :param x: dataset to predict labels of
        :return: numpy array with the probability for each label
        """
        activations = []
        for i in range(len(self.final_weights)):
            if i == 0:
                mul = np.matmul(x, self.final_weights[i]) + self.final_biases[i]
                activations += [1 / (1 + np.exp(-mul))]
            else:
                mul = np.matmul(activations[i - 1], self.final_weights[i]) + self.final_biases[i]
                activations += [1 / (1 + np.exp(-mul))]
        return activations[-1]

    def test(self, x, y, old_model=None, history=None):

        new_output = self.predict_probabilities(x)

        if 'adaboost' not in self.model_name:
            y_new_predicted = np.round(new_output)
            y_new_correct = np.equal(y_new_predicted, y).astype(int)
            accuracy = np.mean(y_new_correct)
            if old_model is None:  # testing pre-update model
                # return sklearn.metrics.roc_auc_score(y, new_output)
                return {'auc': accuracy, 'predicted': y_new_predicted}

        old_output = old_model.predict_probabilities(x)
        y_old_correct = np.equal(np.round(old_output), y).astype(int)

        if 'adaboost' in self.model_name:
            # normalize output to the range [-0.5, 0.5]
            y_new_predicted = old_model.ensemble_weight * (old_output - 0.5) + self.ensemble_weight * (new_output - 0.5)
            # translate sign into a prediction of either 0 or 1
            y_new_predicted = np.greater(y_new_predicted, 0).astype(int)
            y_new_correct = np.equal(y_new_predicted, y).astype(int)
            accuracy = np.mean(y_new_correct)

        compatibility = np.sum(y_old_correct * y_new_correct) / np.sum(y_old_correct)
        return {'compatibility': compatibility, 'auc': accuracy, 'predicted': y_new_predicted}

    def set_hybrid_test(self, history, x, method='stat', layers=None):

        x_train_hist = history.x_train
        y_train_hist = history.y_train
        train_hist_old_output = self.old_model.predict_probabilities(x_train_hist)
        train_hist_old_correct = np.equal(np.round(train_hist_old_output), y_train_hist)
        train_hist_new_output = self.predict_probabilities(x_train_hist)
        train_hist_new_incorrect = np.not_equal(np.round(train_hist_new_output), y_train_hist)

        x_test_hist = history.x_test
        y_test_hist = history.y_test
        test_hist_old_output = self.old_model.predict_probabilities(x_test_hist)
        test_hist_old_correct = np.equal(np.round(test_hist_old_output), y_test_hist)
        test_hist_new_output = self.predict_probabilities(x_test_hist)
        test_hist_new_incorrect = np.not_equal(np.round(test_hist_new_output), y_test_hist)

        if method != 'nn':  # stat or mixed
            train_diss_indexes = (train_hist_old_correct * train_hist_new_incorrect).reshape(-1)
            train_diss_instances = x_train_hist[train_diss_indexes]
            train_diss_labels = y_train_hist[train_diss_indexes]
            train_diss_history = History(train_diss_instances, train_diss_labels, history.width_factor, history.epsilon)
            train_diss_history.set_simple_likelihood(x)
            # dissonant_history.set_simple_likelihood(x, np.abs(self.final_W1.sum(axis=1)))
            self.dissonant_likelihood = train_diss_history.likelihood
        elif method != 'stat':  # neural_network or mixed
            y_train_diss = (train_hist_old_correct * train_hist_new_incorrect).astype(int)
            y_test_diss = (test_hist_old_correct * test_hist_new_incorrect).astype(int)

            # model = Sequential()
            # model.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(l1=0.0, l2=0.1), input_dim=len(x_history[0])))
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # model.fit(x_history, dissonant_y, batch_size=50, epochs=300, verbose=0)
            # self.dissonant_likelihood = model.predict(x)

            version_chooser = NeuralNet(x_train_hist, y_train_diss, x_test_hist, y_test_diss, 128, layers,
                                        'version_nn')

            if method == 'nn':
                self.dissonant_likelihood = version_chooser.predict_probabilities(x)
            self.hybrid_feature_weights = version_chooser.final_weights[0]
            # else:  # mixed
            #     stat_score = np.equal(np.round(model.predict_probabilities(x_history)), y_history).astype(int)
            #     neural_network_score = np.equal(np.round(model.predict_probabilities(x_history)), y_history).astype(int)

        self.dissonant_likelihood_mean = self.dissonant_likelihood.mean()
        self.dissonant_likelihood_std = self.dissonant_likelihood.std()

        self.hybrid_old_output = self.old_model.predict_probabilities(x)
        self.hybrid_new_output = self.predict_probabilities(x)

        if method == 'nn':
            return version_chooser.test(x_train_hist, y_train_diss)

    def hybrid_test(self, y, std_offset):
        # get accuracy and compatibility
        likelihood = self.dissonant_likelihood
        threshold = self.dissonant_likelihood_mean + self.dissonant_likelihood_std * std_offset
        hybrid_output = np.where(likelihood < threshold, self.hybrid_new_output, self.hybrid_old_output)
        predicted = np.round(hybrid_output)
        hybrid_correct = np.equal(predicted, y).astype(int)
        old_correct = np.equal(np.round(self.hybrid_old_output), y).astype(int)
        accuracy = np.mean(hybrid_correct)
        compatibility = np.sum(old_correct * hybrid_correct) / np.sum(old_correct)

        return {'compatibility': compatibility, 'auc': accuracy, 'predicted': predicted}


class History:
    """
    Class that implements the user's history, calculating means and vars.
    """

    def __init__(self, x_train, y_train, x_test, y_test, width_factor=0.01, epsilon=0.0000001):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epsilon = epsilon
        self.width_factor = width_factor
        self.parametric_magnitude_multiplier = 1
        self.kernel_magnitude_multiplier = 1
        self.means = None
        self.norm_const = None
        self.inverse_cov = None
        self.likelihood = None
        self.kernels = None

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
