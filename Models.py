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


class NeuralNetwork:
    """
    Class implementing a Neural Network, capable of training using the log loss + dissonance
    to be able to produce compatible updates.
    """
    def __init__(self, X, Y, train_fraction, train_epochs, batch_size, layers, learning_rate=0.02,
                 diss_weight=None, old_model=None, dissonance_type=None, make_h1_subset=True, copy_h1_weights=True,
                 history=None, use_history=False, initial_stdev=1, make_train_similar_to_history=False,
                 model_type='L0', test_model=True, weights_seed=1, regularization=0.0, plot_train=False,
                 normalize_diss_weight=False):

        self.old_model = old_model
        self.dissonant_likelihood = None

        start_time = int(round(time.time() * 1000))

        # ------------------------------ #
        # PREPARE TRAINING AND TEST SETS #
        # ------------------------------ #

        # shuffle indexes to cover train and test sets
        if old_model is None or not make_h1_subset:
            shuffled_indexes = range(len(X))
            train_stop = train_fraction
            self.train_indexes = shuffled_indexes[:train_stop]
            self.test_indexes = shuffled_indexes[train_stop:]

        else:  # make the old train set to be a subset of the new train set
            shuffled_test_indexes = old_model.test_indexes
            test_stop = train_fraction - len(old_model.train_indexes)
            self.train_indexes = np.concatenate((old_model.train_indexes, shuffled_test_indexes[:test_stop]))
            self.test_indexes = shuffled_test_indexes[test_stop:]

        # assign train and test subsets
        X_train = X[self.train_indexes]
        Y_train = Y[self.train_indexes]
        X_test = X[self.test_indexes]
        Y_test = Y[self.test_indexes]

        if history is not None:
            try:
                kernels_train = history.kernels[self.train_indexes]
                kernels_test = history.kernels[self.test_indexes]
            except TypeError:
                pass
            try:
                likelihood_train = history.likelihood[self.train_indexes]
                likelihood_test = history.likelihood[self.test_indexes]
            except TypeError:
                pass

        # ------------ #
        # CREATE MODEL #
        # ------------ #

        n_features = len(X[0])
        labels_dim = 1

        # these placeholders serve as the input tensors
        x = tf.placeholder(tf.float32, [None, n_features], name='input')
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
                    initial_weights += [tf.truncated_normal([n_features, layers[i]], mean=0,
                                                            stddev=initial_stdev / np.sqrt(n_features), seed=weights_seed)]
                    initial_biases += [tf.truncated_normal([layers[i]], mean=0,
                                                           stddev=initial_stdev / np.sqrt(n_features), seed=weights_seed)]
                else:
                    initial_weights += [tf.random_normal([layers[i-1], layers[i]], mean=0, stddev=initial_stdev,
                                                         seed=weights_seed)]
                    initial_biases += [tf.random_normal([layers[i]], mean=0, stddev=initial_stdev, seed=weights_seed)]

            if len(layers) == 0:
                last_layer = n_features
            else:
                last_layer = layers[-1]
            initial_weights += [tf.random_normal([last_layer, labels_dim], mean=0, stddev=initial_stdev,
                                                 seed=weights_seed)]
            initial_biases += [tf.random_normal([labels_dim], mean=0, stddev=initial_stdev, seed=weights_seed)]
        else:
            for i in range(len(layers)):
                initial_weights += [tf.convert_to_tensor(old_model.final_weights[i])]
                initial_biases += [tf.convert_to_tensor(old_model.final_biases[i])]

        # build layers
        weights = []
        biases = []
        activations = []
        for i in range(len(layers)):
            weights += [tf.Variable(initial_weights[i], name='weights_'+str(i+1))]
            biases += [tf.Variable(initial_biases[i], name='biases_'+str(i+1))]
            if i == 0:
                activations += [tf.sigmoid((tf.matmul(x, weights[i]) + biases[i]), name='activations_'+str(i+1))]
            else:
                activations += [tf.sigmoid((tf.matmul(activations[i-1], weights[i]) + biases[i]), name='activations_' + str(i + 1))]

        weights += [tf.Variable(initial_weights[-1], name='weights_' + str(len(layers) + 1))]
        biases += [tf.Variable(initial_biases[-1], name='biases_' + str(len(layers) + 1))]

        if len(layers) == 0:
            logits = tf.matmul(x, weights[-1]) + biases[-1]
        else:
            logits = tf.matmul(activations[-1], weights[-1]) + biases[-1]
        output = tf.nn.sigmoid(logits, name='output')

        # for non parametric compatibility
        if model_type in ['L1', 'L2', 'L3']:
            kernels = tf.placeholder(tf.float32, [None, len(history.instances)], name='kernels')
            hist_x = tf.placeholder(tf.float32, [None, n_features], name='hist_input')
            hist_y = tf.placeholder(tf.float32, [None, labels_dim], name='hist_labels')
            hist_y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='hist_old_corrects')

            hist_activations = []
            for i in range(len(layers)):
                if i == 0:
                    hist_activations += [tf.sigmoid((tf.matmul(hist_x, weights[i]) + biases[i]), name='hist_activations_'+str(i+1))]
                else:
                    hist_activations += [tf.sigmoid((tf.matmul(hist_activations[i-1], weights[i]) + biases[i]), name='hist_activations_'+str(i+1))]

            if len(layers) == 0:
                hist_logits = tf.matmul(hist_x, weights[-1]) + biases[-1]
            else:
                hist_logits = tf.matmul(hist_activations[-1], weights[-1]) + biases[-1]

        # model evaluation tensors
        correct_prediction = tf.equal(tf.round(output), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        y_new_correct = tf.cast(tf.equal(tf.round(output), y), tf.float32)

        # loss computation
        log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        # log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

        # dissonance computation
        if old_model is None:
            loss = log_loss
        else:
            if dissonance_type == "D":
                dissonance = y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            elif dissonance_type == "D'":
                dissonance = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_old_probabilities, logits=logits)
            elif dissonance_type == "D''":
                dissonance = y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_old_probabilities,
                                                                                     logits=logits)
            else:
                raise Exception("invalid dissonance type")

            if history is None:
                if normalize_diss_weight:
                    loss = (1-diss_weight) * log_loss + diss_weight * tf.reduce_mean(dissonance)
                else:
                    loss = log_loss + diss_weight * dissonance
                    # loss = log_loss + diss_weight * tf.reduce_mean(dissonance)
                compatibility = tf.reduce_sum(y_old_correct * y_new_correct) / tf.reduce_sum(y_old_correct)
            else:
                if not use_history:
                    if normalize_diss_weight:
                        loss = (1-diss_weight) * log_loss + diss_weight * tf.reduce_mean(dissonance)
                    else:
                        loss = log_loss + diss_weight * dissonance
                        # loss = log_loss + diss_weight * tf.reduce_mean(dissonance)
                else:
                    if model_type == 'L2':
                        # shape = tf.shape(kernels)
                        kernel_likelihood = tf.reduce_sum(kernels, axis=1) / len(history.instances)
                        if normalize_diss_weight:
                            loss = (1 - diss_weight) * log_loss + diss_weight * kernel_likelihood * tf.reduce_mean(
                                dissonance)
                        else:
                            loss = log_loss + diss_weight * kernel_likelihood * dissonance
                            # loss = log_loss + diss_weight * kernel_likelihood * tf.reduce_mean(dissonance)

                    elif model_type == 'L1' or model_type == 'L3':
                        hist_dissonance = hist_y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=hist_y,
                            logits=hist_logits,
                            name='hist_dissonance')
                        hist_dissonance = tf.reshape(hist_dissonance, [-1])
                        reduced_diss_weight = diss_weight / 1

                        if model_type == 'L1':
                            kernel_likelihood = tf.reduce_sum(kernels * hist_dissonance) / tf.reduce_sum(kernels)
                            if normalize_diss_weight:
                                loss = (1-diss_weight) * log_loss + diss_weight * tf.reduce_mean(kernel_likelihood)
                            else:
                                loss = log_loss + reduced_diss_weight * kernel_likelihood
                                # loss = log_loss + reduced_diss_weight * tf.reduce_mean(kernel_likelihood)

                        else:  # L3
                            if normalize_diss_weight:
                                loss = (1 - diss_weight) * log_loss + diss_weight * tf.reduce_mean(hist_dissonance)
                            else:
                                loss = log_loss + reduced_diss_weight * hist_dissonance
                                # loss = log_loss + reduced_diss_weight * tf.reduce_mean(hist_dissonance)

                    else:  # L0
                        if normalize_diss_weight:
                            loss = (1-diss_weight) * log_loss + diss_weight * tf.reduce_mean(dissonance * likelihood)
                        else:
                            loss = log_loss + diss_weight * dissonance * likelihood
                            # loss = log_loss + diss_weight * tf.reduce_mean(dissonance * likelihood)

                if model_type in ['L1', 'L2']:
                    compatibility = tf.reduce_sum(
                        y_old_correct * y_new_correct * tf.reduce_sum(kernels, axis=1)) / tf.reduce_sum(
                        y_old_correct * tf.reduce_sum(kernels, axis=1))
                else:
                    compatibility = tf.reduce_sum(y_old_correct * y_new_correct * likelihood) / tf.reduce_sum(
                        y_old_correct * likelihood)

        # loss = tf.reduce_mean(loss)

        if regularization > 0:
            regularizers = tf.nn.l2_loss(weights[0])
            for i in range(1, len(weights)):
                regularizers += tf.nn.l2_loss(weights[i])
            loss = tf.reduce_mean(loss + regularization * regularizers)

        # prepare training
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) (obsolete)
        train_step = tf.train.AdamOptimizer().minimize(loss)
        init_op = tf.global_variables_initializer()

        # ----------- #
        # TRAIN MODEL #
        # ----------- #

        with tf.Session() as sess:
            sess.run(init_op)
            batches = int(len(X_train) / batch_size)

            if old_model is None:  # without compatibility
                # print("h1 training: train size = " + str(train_fraction))
                # print("\nTRAINING h1:\ntrain fraction = " + str(int(100 * train_fraction)) + "%\n")

                plot_x = []
                self.plot_train_losses = []
                self.plot_train_accuracy = []
                self.plot_test_losses = []
                self.plot_test_accuracy = []

                for epoch in range(train_epochs):
                    # if epoch % 10 == 0:
                    #     sys.stdout.write("epoch %d/%d\r" % (epoch+1, train_epochs))
                    #     sys.stdout.flush()
                    # losses = 0
                    # accs = 0
                    for batch in range(batches + 1):

                        batch_start = batch * batch_size
                        if batch_start == X_train.shape[0]:
                            continue  # in case the len of train set is a multiple of batch size
                        batch_end = min((batch + 1) * batch_size, X_train.shape[0])
                        X_batch = X_train[batch_start:batch_end]
                        Y_batch = Y_train[batch_start:batch_end]

                        sess.run(train_step, feed_dict={x: X_batch, y: Y_batch})

                    if plot_train:
                        # print(str(epoch + 1) + "/" + str(train_epochs) + "\tloss = %.4f" %losses +"\tauc = %.4f" % self.acc)
                        plot_x += [epoch]

                        acc, lss = sess.run([accuracy, loss], feed_dict={x: X_train, y: Y_train})
                        self.plot_train_losses += [lss]
                        self.plot_train_accuracy += [acc]

                        acc, lss = sess.run([accuracy, loss], feed_dict={x: X_test, y: Y_test})
                        self.plot_test_losses += [lss]
                        self.plot_test_accuracy += [acc]

                # if plot_train:
                #     # plt.plot(plot_x, plot_train_losses, 'b', label='loss')
                #     plt.plot(plot_x, plot_train_accuracy, label='train accuracy')
                #     plt.plot(plot_x, plot_test_accuracy, label='test accuracy')
                #     plt.xlabel('epoch')
                #     plt.ylabel('accuracy')
                #     plt.legend()
                #     plt.grid()
                #     runtime = int((round(time.time() * 1000)) - start_time) / 60000
                #     plt.title('seed='+str(seed)+' train='+str(len(Y_train))+' test='+str(len(Y_test))+
                #               ' epochs='+str(train_epochs)+' run=%.2f min' % runtime+'\nlayers='+str(layers)
                #               +' reg='+str(regularization))
                #     plt.savefig(plots_dir + '\\model_training\\'+'h1_seed_'+str(seed))
                #     plt.show()
                #     plt.clf()

                if test_model:
                    acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_test, y: Y_test})
                    # self.auc = sklearn.metrics.roc_auc_score(Y_test, out)
                    self.accuracy = acc


                    print("test acc = %.4f" % self.accuracy)

            else:  # with compatibility

                if history is not None:
                    if use_history:
                        history_string = "MODEL " + model_type
                    else:
                        history_string = "IGNORE HISTORY"
                else:
                    history_string = "MODEL NO HISTORY"

                # print(history_string+", train size = " + str(train_fraction) + ", diss weight = " + str(diss_weight))

                # get the old model predictions
                Y_train_old_probabilities = old_model.predict_probabilities(X_train)
                Y_train_old_labels = tf.round(Y_train_old_probabilities).eval()
                Y_train_old_correct = tf.cast(tf.equal(Y_train_old_labels, Y_train), tf.float32).eval()
                Y_test_old_probabilities = old_model.predict_probabilities(X_test)
                Y_test_old_labels = tf.round(Y_test_old_probabilities).eval()
                Y_test_old_correct = tf.cast(tf.equal(Y_test_old_labels, Y_test), tf.float32).eval()

                if model_type in ['L1', 'L2', 'L3']:
                    hist_Y_old_probabilities = old_model.predict_probabilities(history.instances)
                    hist_Y_old_labels = tf.round(hist_Y_old_probabilities).eval()
                    hist_Y_old_correct = tf.cast(tf.equal(hist_Y_old_labels, history.labels), tf.float32).eval()

                for epoch in range(train_epochs):
                    # if epoch % 10 == 0:
                    #     sys.stdout.write("epoch %d/%d\r" % (epoch + 1, train_epochs))
                    #     sys.stdout.flush()

                    for batch in range(batches):
                        batch_start = batch * batch_size
                        if batch_start == X_train.shape[0]:
                            continue  # in case the len of train set is a multiple of batch size
                        batch_end = min((batch + 1) * batch_size, X_train.shape[0])
                        X_batch = X_train[batch_start:batch_end]
                        Y_batch = Y_train[batch_start:batch_end]
                        Y_batch_old_probabilities = Y_train_old_probabilities[batch_start:batch_end]
                        Y_batch_old_correct = Y_train_old_correct[batch_start:batch_end]

                        # train the new model, and then get the accuracy and loss from it
                        if history is None:
                            sess.run(train_step,
                                     feed_dict={x: X_batch, y: Y_batch,
                                                y_old_probabilities: Y_batch_old_probabilities,
                                                y_old_correct: Y_batch_old_correct})
                        else:
                            if model_type in ['L1', 'L2']:
                                kernels_batch = kernels_train[batch_start:batch_end]
                                sess.run(train_step,
                                         feed_dict={x: X_batch, y: Y_batch,
                                                    y_old_probabilities: Y_batch_old_probabilities,
                                                    y_old_correct: Y_batch_old_correct,
                                                    hist_x: history.instances,
                                                    hist_y: history.labels,
                                                    hist_y_old_correct: hist_Y_old_correct,
                                                    kernels: kernels_batch})
                            elif model_type == 'L0':
                                likelihood_batch = likelihood_train[batch_start:batch_end]
                                sess.run(train_step,
                                         feed_dict={x: X_batch, y: Y_batch,
                                                    y_old_probabilities: Y_batch_old_probabilities,
                                                    y_old_correct: Y_batch_old_correct,
                                                    likelihood: likelihood_batch})
                            else:  # model == 'L3'
                                sess.run(train_step,
                                         feed_dict={x: X_batch, y: Y_batch,
                                                    y_old_probabilities: Y_batch_old_probabilities,
                                                    y_old_correct: Y_batch_old_correct,
                                                    hist_x: history.instances,
                                                    hist_y: history.labels,
                                                    hist_y_old_correct: hist_Y_old_correct})

                if test_model:
                    if history is None:
                        out, com, new_correct, acc = sess.run(
                            [output, compatibility, y_new_correct, accuracy],
                            feed_dict={x: X_test, y: Y_test,
                                       y_old_probabilities: Y_test_old_probabilities,
                                       y_old_correct: Y_test_old_correct})
                    else:
                        if model_type in ['L1', 'L2']:
                            if use_history:
                                out, com, new_correct, _hist_diss, _kernel_likelihood, acc = sess.run(
                                    [output, compatibility, y_new_correct, hist_dissonance, kernel_likelihood, accuracy],
                                    feed_dict={x: X_test, y: Y_test,
                                               y_old_probabilities: Y_test_old_probabilities,
                                               y_old_correct: Y_test_old_correct,
                                               hist_x: history.instances,
                                               hist_y: history.labels,
                                               hist_y_old_correct: hist_Y_old_correct,
                                               kernels: kernels_test})
                            else:
                                out, com, new_correct, acc = sess.run(
                                    [output, compatibility, y_new_correct, accuracy],
                                    feed_dict={x: X_test, y: Y_test,
                                               y_old_probabilities: Y_test_old_probabilities,
                                               y_old_correct: Y_test_old_correct,
                                               # hist_y: history.labels,
                                               # hist_y_old_correct: hist_Y_old_correct,
                                               kernels: kernels_test})
                        else:
                            out, com, new_correct, acc = sess.run(
                                [output, compatibility, y_new_correct, accuracy],
                                feed_dict={x: X_test, y: Y_test,
                                           y_old_probabilities: Y_test_old_probabilities,
                                           y_old_correct: Y_test_old_correct,
                                           likelihood: likelihood_test})

                    self.compatibility = com
                    # self.auc = sklearn.metrics.roc_auc_score(Y_test, out)
                    self.accuracy = acc
                    self.new_correct = new_correct
                    self.old_correct = Y_test_old_correct

                    # print("FINISHED:\ttest auc = %.4f" % self.auc + ", compatibility = %.4f" % self.compatibility)
                    print("test acc = %.4f" % self.accuracy + ", compatibility = %.4f" % self.compatibility)
                    # print("log loss = "+str(np.sum(log_lss)))
                    # print("dissonance = "+str(np.sum(diss)))

            # save weights
            self.final_weights = []
            self.final_biases = []
            for i in range(len(weights)):
                self.final_weights += [weights[i].eval()]
                self.final_biases += [biases[i].eval()]

            runtime = str(int((round(time.time() * 1000)) - start_time) / 1000)
            print("runtime = " + str(runtime) + " secs\n")

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
                mul = np.matmul(activations[i-1], self.final_weights[i]) + self.final_biases[i]
                activations += [1 / (1 + np.exp(-mul))]
        return activations[-1]

    def test(self, x, y, old_model=None, history=None):
        new_output = self.predict_probabilities(x)
        predicted = np.round(new_output)
        y_new_correct = np.equal(predicted, y).astype(int)
        accuracy = np.mean(y_new_correct)

        if old_model is None:
            # return sklearn.metrics.roc_auc_score(y, new_output)
            return {'auc': accuracy, 'predicted': predicted}

        old_output = old_model.predict_probabilities(x)
        y_old_correct = np.equal(np.round(old_output), y).astype(int)

        # if history is not None:
        #     likelihood = history.likelihood
        #     compatibility = np.sum(y_old_correct * y_new_correct * likelihood) / np.sum(y_old_correct * likelihood)
        #
        # else:
        compatibility = np.sum(y_old_correct * y_new_correct) / np.sum(y_old_correct)
        # return {'compatibility': compatibility.eval(), 'auc': sklearn.metrics.roc_auc_score(y, new_output)}
        return {'compatibility': compatibility, 'auc': accuracy, 'predicted': predicted}

    def set_hybrid_test(self, history, x, method='stat', layers=None):

        x_history = history.instances
        y_history = history.labels
        hist_old_output = self.old_model.predict_probabilities(x_history)
        hist_old_correct = np.equal(np.round(hist_old_output), y_history)
        hist_new_output = self.predict_probabilities(x_history)
        hist_new_incorrect = np.not_equal(np.round(hist_new_output), y_history)

        if method != 'nn':  # stat or mixed
            dissonant_indexes = (hist_old_correct * hist_new_incorrect).reshape(-1)
            dissonant_instances = x_history[dissonant_indexes]
            dissonant_labels = y_history[dissonant_indexes]
            dissonant_history = History(dissonant_instances, dissonant_labels, history.width_factor, history.epsilon)
            dissonant_history.set_simple_likelihood(x)
            # dissonant_history.set_simple_likelihood(x, np.abs(self.final_W1.sum(axis=1)))
            self.dissonant_likelihood = dissonant_history.likelihood
        elif method != 'stat':  # neural_network or mixed
            y_dissonant = (hist_old_correct * hist_new_incorrect).astype(int)

            # model = Sequential()
            # model.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(l1=0.0, l2=0.1), input_dim=len(x_history[0])))
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # model.fit(x_history, dissonant_y, batch_size=50, epochs=300, verbose=0)
            # self.dissonant_likelihood = model.predict(x)

            model = NeuralNetwork(x_history, y_dissonant, len(x_history), 200, 128, layers, weights_seed=1)
            tf.reset_default_graph()

            if method == 'nn':
                self.dissonant_likelihood = model.predict_probabilities(x)
            # else:  # mixed
            #     stat_score = np.equal(np.round(model.predict_probabilities(x_history)), y_history).astype(int)
            #     neural_network_score = np.equal(np.round(model.predict_probabilities(x_history)), y_history).astype(int)

        self.dissonant_likelihood_mean = self.dissonant_likelihood.mean()
        self.dissonant_likelihood_std = self.dissonant_likelihood.std()

        self.hybrid_old_output = self.old_model.predict_probabilities(x)
        self.hybrid_new_output = self.predict_probabilities(x)

        if method == 'nn':
            return model.test(x_history, y_dissonant)

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

    def __init__(self, instances, labels=None, width_factor=0.1, epsilon=0.0000001):
        self.instances = instances
        self.labels = labels
        self.means = np.mean(instances, axis=0)
        self.vars = np.var(instances, axis=0) * width_factor + epsilon
        self.epsilon = epsilon
        self.width_factor = width_factor
        self.likelihood = None
        self.kernels = None
        self.parametric_magnitude_multiplier = 1
        self.kernel_magnitude_multiplier = 1

    def set_simple_likelihood(self, df, weights=None, magnitude_multiplier=1):
        # compute likelihood for each attribute
        self.parametric_magnitude_multiplier = magnitude_multiplier
        diff = np.subtract(df, self.means)
        sqr_diff = np.power(diff, 2)
        div = np.add(sqr_diff, self.vars)
        attribute_likelihoods = np.divide(self.vars, div) * magnitude_multiplier

        # todo: experimenting with likelihood here
        # merge the likelihood of all attributes
        if weights is None:
            self.likelihood = np.mean(attribute_likelihoods, axis=1)
        else:
            self.likelihood = np.average(attribute_likelihoods, axis=1, weights=weights)
        # self.likelihood = np.round(self.likelihood)
        # self.likelihood = 1 + self.likelihood
        self.likelihood = np.reshape(self.likelihood, (len(df), 1))

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
            for hist_instance in self.instances:
                entry += [np.linalg.norm(instance - hist_instance)]
            distances += [entry]
        distances = np.asanyarray(distances)
        self.kernels = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
            -1 / 2 * np.square(distances / sigma)) * magnitude_multiplier
