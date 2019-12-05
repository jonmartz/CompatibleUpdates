import csv
import os.path
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import auc
# import tensorflow as tf
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import L1L2
import seaborn as sn


class MLP:
    """
    Class implementing a Multi Layered Perceptron, capable of training using the log loss + dissonance
    to be able to produce compatible updates.
    """

    def __init__(self, X, Y, train_fraction, train_epochs, batch_size, layer_1_neurons, layer_2_neurons, learning_rate=0.02,
                 diss_weight=None, old_model=None, dissonance_type=None, make_h1_subset=True, copy_h1_weights=True,
                 history=None, use_history=False, train_start=0, initial_stdev=1, make_train_similar_to_history=False,
                 model_type='L0', test_model=True, weights_seed=1, regularizer_weight=0.0):

        self.old_model = old_model
        self.dissonant_likelihood = None

        start_time = int(round(time.time() * 1000))

        # ------------------------------ #
        # PREPARE TRAINING AND TEST SETS #
        # ------------------------------ #

        # shuffle indexes to cover train and test sets
        if old_model is None or not make_h1_subset:
            # shuffled_indexes = np.random.randint(len(X), size=len(X))
            shuffled_indexes = range(len(X))
            # train_stop = int(len(X) * train_fraction)
            train_stop = train_fraction
            self.train_indexes = shuffled_indexes[:train_stop]
            self.test_indexes = shuffled_indexes[train_stop:]

            # # todo: trying to eliminate randomness here
            # indexes = range(len(X))
            # train_size = int(len(X) * train_fraction)
            # self.train_indexes = indexes[train_start * train_size:(train_start + 1) * train_size]

            # if make_train_similar_to_history:
            #     similar_indexes = []
            #     for index in self.train_indexes:
            #         # if history.likelihood[index] == 0:
            #         if history.likelihood[index] == 1:
            #             similar_indexes += [index]
            #     self.train_indexes = similar_indexes

            # self.test_indexes = [x for x in indexes if x not in self.train_indexes]

        else:  # make the old train set to be a subset of the new train set
            # shuffled = np.random.randint(len(old_model.test_indexes), size=len(old_model.test_indexes))
            # shuffled_test_indexes = old_model.test_indexes[shuffled]

            # todo: trying to eliminate randomness here
            shuffled_test_indexes = old_model.test_indexes
            # test_stop = int(len(X) * train_fraction - len(old_model.train_indexes))
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
        if old_model is None or not copy_h1_weights:
            # if True:
            w1_initial = tf.truncated_normal([n_features, layer_1_neurons], mean=0,stddev=initial_stdev / np.sqrt(n_features), seed=weights_seed)
            b1_initial = tf.truncated_normal([layer_1_neurons], mean=0, stddev=initial_stdev / np.sqrt(n_features), seed=weights_seed)
            w2_initial = tf.random_normal([layer_1_neurons, layer_2_neurons], mean=0, stddev=initial_stdev, seed=weights_seed)
            b2_initial = tf.random_normal([layer_2_neurons], mean=0, stddev=initial_stdev, seed=weights_seed)
            wo_initial = tf.random_normal([layer_2_neurons, labels_dim], mean=0, stddev=initial_stdev, seed=weights_seed)
            bo_initial = tf.random_normal([labels_dim], mean=0, stddev=initial_stdev, seed=weights_seed)
        else:
            w1_initial = tf.convert_to_tensor(old_model.final_W1)
            b1_initial = tf.convert_to_tensor(old_model.final_b1)
            w2_initial = tf.convert_to_tensor(old_model.final_W2)
            b2_initial = tf.convert_to_tensor(old_model.final_b2)
            wo_initial = tf.convert_to_tensor(old_model.final_Wo)
            bo_initial = tf.convert_to_tensor(old_model.final_bo)

        # first layer
        W1 = tf.Variable(w1_initial, name='weights1')
        b1 = tf.Variable(b1_initial, name='biases1')
        y1 = tf.sigmoid((tf.matmul(x, W1) + b1), name='activationLayer1')

        # second layer
        W2 = tf.Variable(w2_initial, name='weights2')
        b2 = tf.Variable(b2_initial, name='biases2')
        y2 = tf.sigmoid((tf.matmul(y1, W2) + b2), name='activationLayer2')

        # output layer
        Wo = tf.Variable(wo_initial, name='weightsOut')
        bo = tf.Variable(bo_initial, name='biasesOut')
        logits = tf.matmul(y2, Wo) + bo
        output = tf.nn.sigmoid(logits, name='activationOutputLayer')

        # for non parametric compatibility
        if model_type in ['L1', 'L2']:
            kernels = tf.placeholder(tf.float32, [None, len(history.instances)], name='kernels')
            hist_x = tf.placeholder(tf.float32, [None, n_features], name='hist_input')
            hist_y = tf.placeholder(tf.float32, [None, labels_dim], name='hist_labels')
            hist_y_old_correct = tf.placeholder(tf.float32, [None, labels_dim], name='hist_old_corrects')

            hist_y1 = tf.sigmoid((tf.matmul(hist_x, W1) + b1), name='hist_activationLayer1')
            hist_y2 = tf.sigmoid((tf.matmul(hist_y1, W2) + b2), name='hist_activationLayer2')
            hist_logits = tf.matmul(hist_y2, Wo) + bo

        # model evaluation tensors
        correct_prediction = tf.equal(tf.round(output), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        y_new_correct = tf.cast(tf.equal(tf.round(output), y), tf.float32)

        # loss computation
        log_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)

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
                loss = log_loss + diss_weight * dissonance
                compatibility = tf.reduce_sum(y_old_correct * y_new_correct) / tf.reduce_sum(y_old_correct)
            else:
                if not use_history:
                    loss = log_loss + diss_weight * dissonance
                else:
                    if model_type in ['L1', 'L2']:

                        # product = kernels * hist_dissonance
                        # numerator = tf.reduce_sum(product, axis=1)
                        # denominator = tf.reduce_sum(kernels, axis=1)
                        # kernel_likelihood = numerator / denominator

                        if model_type == 'L1':
                            hist_dissonance = hist_y_old_correct * tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=hist_y,
                                logits=hist_logits,
                                name='hist_dissonance')
                            hist_dissonance = tf.reshape(hist_dissonance, [-1])
                            kernel_likelihood = tf.reduce_sum(kernels * hist_dissonance) / tf.reduce_sum(kernels)
                            loss = log_loss + diss_weight * kernel_likelihood
                        elif model_type == 'L2':
                            # shape = tf.shape(kernels)
                            kernel_likelihood = tf.reduce_sum(kernels, axis=1) / len(history.instances)
                            loss = log_loss + diss_weight * kernel_likelihood * dissonance
                    else:
                        loss = log_loss + diss_weight * dissonance * likelihood

                if model_type in ['L1', 'L2']:
                    # todo: maybe use average instead of sum?
                    compatibility = tf.reduce_sum(
                        y_old_correct * y_new_correct * tf.reduce_sum(kernels, axis=1)) / tf.reduce_sum(
                        y_old_correct * tf.reduce_sum(kernels, axis=1))
                else:
                    compatibility = tf.reduce_sum(y_old_correct * y_new_correct * likelihood) / tf.reduce_sum(
                        y_old_correct * likelihood)

        if regularizer_weight == 0:
            loss = tf.reduce_mean(loss)
        else:
            regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(Wo)
            loss = tf.reduce_mean(loss + regularizer_weight * regularizer)

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
                print("TRAINING h1: train size = " + str(train_fraction))
                # print("\nTRAINING h1:\ntrain fraction = " + str(int(100 * train_fraction)) + "%\n")

                plot_x = []
                plot_losses = []
                plot_auc = []
                for epoch in range(train_epochs):
                    losses = 0
                    accs = 0
                    for batch in range(batches + 1):
                        # shuffled_indexes = np.random.randint(X_train.shape[0], size=batch_size)
                        # X_batch = X_train[shuffled_indexes]
                        # Y_batch = Y_train[shuffled_indexes]

                        # todo: trying to eliminate randomness here
                        batch_start = batch * batch_size
                        if batch_start == X_train.shape[0]:
                            continue  # in case the len of train set is a multiple of batch size
                        batch_end = min((batch + 1) * batch_size, X_train.shape[0])
                        X_batch = X_train[batch_start:batch_end]
                        Y_batch = Y_train[batch_start:batch_end]

                        # train the model, and then get the accuracy and loss from model
                        sess.run(train_step, feed_dict={x: X_batch, y: Y_batch})
                        # acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_batch, y: Y_batch})
                        # losses = losses + np.sum(lss)
                        # accs = accs + np.sum(acc)

                if test_model:
                    acc, lss, out = sess.run([accuracy, loss, output], feed_dict={x: X_test, y: Y_test})
                    # self.auc = sklearn.metrics.roc_auc_score(Y_test, out)
                    self.accuracy = acc

                    # print(str(epoch + 1) + "/" + str(train_epochs) + "\tloss = %.4f" %losses +"\tauc = %.4f" % self.auc)

                    #     plot_x += [epoch]
                    #     plot_losses += [losses]
                    #     plot_auc += [ self.auc]
                    #
                    # plt.plot(plot_x, plot_losses, 'bs', label='loss')
                    # plt.plot(plot_x, plot_auc, 'r^', label='auc')
                    # plt.xlabel('epoch')
                    # plt.legend(('loss', 'auc'),loc='center left')
                    # plt.title('Training')
                    # plt.show()
                    # plt.clf()

                    # print("test auc = %.4f" % self.auc)
                    print("test acc = %.4f" % self.accuracy)

            else:  # with compatibility

                if history is not None:
                    if use_history:
                        history_string = "USING " + model_type
                    else:
                        history_string = "IGNORE HISTORY"
                else:
                    history_string = "NO HISTORY"

                print("TRAINING h2: train size = " + str(train_fraction) + ", diss weight = " + str(diss_weight)
                      # str(int(100 * train_fraction)) + "%, diss weight = " + str(diss_weight)
                      + ", diss type = " + str(dissonance_type) + ", " + history_string)

                # get the old model predictions
                Y_train_old_probabilities = old_model.predict_probabilities(X_train)
                Y_train_old_labels = tf.round(Y_train_old_probabilities).eval()
                Y_train_old_correct = tf.cast(tf.equal(Y_train_old_labels, Y_train), tf.float32).eval()
                Y_test_old_probabilities = old_model.predict_probabilities(X_test)
                Y_test_old_labels = tf.round(Y_test_old_probabilities).eval()
                Y_test_old_correct = tf.cast(tf.equal(Y_test_old_labels, Y_test), tf.float32).eval()

                if model_type in ['L1', 'L2']:
                    hist_Y_old_probabilities = old_model.predict_probabilities(history.instances)
                    # hist_Y_old_probabilities = tf.nn.sigmoid(hist_Y_old_logits, name='hist_probabilities').eval()
                    hist_Y_old_labels = tf.round(hist_Y_old_probabilities).eval()
                    hist_Y_old_correct = tf.cast(tf.equal(hist_Y_old_labels, history.labels), tf.float32).eval()

                for epoch in range(train_epochs):
                    for batch in range(batches):
                        # shuffled_indexes = np.random.randint(X_train.shape[0], size=batch_size)
                        # X_batch = X_train[shuffled_indexes]
                        # Y_batch = Y_train[shuffled_indexes]
                        # Y_batch_old_labels = Y_train_old_labels[shuffled_indexes]
                        # Y_batch_old_correct = Y_train_old_correct[shuffled_indexes]

                        # todo: trying to eliminate randomness here
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
                            else:
                                likelihood_batch = likelihood_train[batch_start:batch_end]
                                sess.run(train_step,
                                         feed_dict={x: X_batch, y: Y_batch,
                                                    y_old_probabilities: Y_batch_old_probabilities,
                                                    y_old_correct: Y_batch_old_correct,
                                                    likelihood: likelihood_batch})

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
            self.final_W1 = W1.eval()
            self.final_b1 = b1.eval()
            self.final_W2 = W2.eval()
            self.final_b2 = b2.eval()
            self.final_Wo = Wo.eval()
            self.final_bo = bo.eval()

            runtime = str(int((round(time.time() * 1000)) - start_time) / 1000)
            print("runtime = " + str(runtime) + " secs\n")

    def predict_probabilities(self, x):
        """
        predict the labels for dataset x
        :param x: dataset to predict labels of
        :return: numpy array with the probability for each label
        """
        # _x = tf.cast(x, tf.float32).eval()
        # _y1 = tf.sigmoid((tf.matmul(_x, self.final_W1) + self.final_b1)).eval()
        # _y2 = tf.sigmoid((tf.matmul(_y1, self.final_W2) + self.final_b2)).eval()
        # _logits = (tf.matmul(_y2, self.final_Wo) + self.final_bo).eval()
        # _out = tf.nn.sigmoid(_logits, name='activationOutputLayer').eval()

        mul1 = np.matmul(x, self.final_W1) + self.final_b1
        y1 = 1 / (1 + np.exp(-mul1))
        mul2 = np.matmul(y1, self.final_W2) + self.final_b2
        y2 = 1 / (1 + np.exp(-mul2))
        logits = np.matmul(y2, self.final_Wo) + self.final_bo
        return 1 / (1 + np.exp(-logits))

    def test(self, x, y, old_model=None, history=None):
        new_output = self.predict_probabilities(x)
        y_new_correct = np.equal(np.round(new_output), y).astype(int)
        accuracy = np.mean(y_new_correct)

        if old_model is None:
            # return sklearn.metrics.roc_auc_score(y, new_output)
            return accuracy

        old_output = old_model.predict_probabilities(x)
        y_old_correct = np.equal(np.round(old_output), y).astype(int)

        # if history is not None:
        #     likelihood = history.likelihood
        #     compatibility = np.sum(y_old_correct * y_new_correct * likelihood) / np.sum(y_old_correct * likelihood)
        #
        # else:
        compatibility = np.sum(y_old_correct * y_new_correct) / np.sum(y_old_correct)
        # return {'compatibility': compatibility.eval(), 'auc': sklearn.metrics.roc_auc_score(y, new_output)}
        return {'compatibility': compatibility, 'auc': accuracy}

    def set_hybrid_test(self, history, x, y, method='parametric'):

        x_history = history.instances
        y_history = history.labels
        hist_old_output = self.old_model.predict_probabilities(x_history)
        hist_old_correct = np.equal(np.round(hist_old_output), y_history)
        hist_new_output = self.predict_probabilities(x_history)
        hist_new_incorrect = np.not_equal(np.round(hist_new_output), y_history)

        if method != 'neural_network':  # parametric or mixed
            dissonant_indexes = (hist_old_correct * hist_new_incorrect).reshape(-1)
            dissonant_instances = x_history[dissonant_indexes]
            dissonant_labels = y_history[dissonant_indexes]
            dissonant_history = History(dissonant_instances, dissonant_labels, history.width_factor, history.epsilon)
            dissonant_history.set_simple_likelihood(x)
            # dissonant_history.set_simple_likelihood(x, np.abs(self.final_W1.sum(axis=1)))
            self.dissonant_likelihood = dissonant_history.likelihood
        elif method != 'parametric':  # neural_network or mixed
            y_dissonant = (hist_old_correct * hist_new_incorrect).astype(int)

            # model = Sequential()
            # model.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(l1=0.0, l2=0.1), input_dim=len(x_history[0])))
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # model.fit(x_history, dissonant_y, batch_size=50, epochs=300, verbose=0)
            # self.dissonant_likelihood = model.predict(x)

            model = MLP(x_history, y_dissonant, len(x_history), 200, batch_size, layer_1_size, layer_2_size, weights_seed=1)
            tf.reset_default_graph()

            if method == 'neural_network':
                self.dissonant_likelihood = model.predict_probabilities(x)
            # else:  # mixed
            #     parametric_score = np.equal(np.round(model.predict_probabilities(x_history)), y_history).astype(int)
            #     neural_network_score = np.equal(np.round(model.predict_probabilities(x_history)), y_history).astype(int)

        self.dissonant_likelihood_mean = self.dissonant_likelihood.mean()
        self.dissonant_likelihood_std = self.dissonant_likelihood.std()

        self.hybrid_old_output = self.old_model.predict_probabilities(x)
        self.hybrid_new_output = self.predict_probabilities(x)

        if method == 'neural_network':
            return model.test(x_history, y_dissonant)

    def hybrid_test(self, y, std_offset):
        # get accuracy and compatibility
        likelihood = self.dissonant_likelihood
        threshold = self.dissonant_likelihood_mean + self.dissonant_likelihood_std * std_offset
        hybrid_output = np.where(likelihood < threshold, self.hybrid_new_output, self.hybrid_old_output)
        hybrid_correct = np.equal(np.round(hybrid_output), y).astype(int)
        old_correct = np.equal(np.round(self.hybrid_old_output), y).astype(int)
        accuracy = np.mean(hybrid_correct)
        compatibility = np.sum(old_correct * hybrid_correct) / np.sum(old_correct)

        return {'compatibility': compatibility, 'auc': accuracy}


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


def make_monotonic(x, y):
    i = 0
    while i < len(x)-1:
        if x[i + 1] < x[i]:
            del x[i + 1]
            del y[i + 1]
        else:
            i += 1


def plot_confusion_matrix(model, x, y, title, path, plot_confusion=True):
    if not plot_confusion:
        return
    h1_confusion_matrix = confusion_matrix(y, np.round(model.predict_probabilities(x)))
    df_confusion_matrix = pd.DataFrame(h1_confusion_matrix, ['true 0', 'true 1'], ['predicted 0', 'predicted 1'])
    sn.set(font_scale=2.0)
    sn.heatmap(df_confusion_matrix, annot=True, cbar=False, cmap="YlGnBu", fmt="d")
    plt.title(title)
    plt.savefig(path)
    # plt.show()
    plt.clf()


# Data-set paths

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\creditRiskAssessment\\heloc_dataset_v1.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\creditRiskAssessment.csv"
# target_col = 'RiskPerformance'
# dataset_fraction = 0.5
# threshold = 75
# users = {'1': df[:100].loc[df['ExternalRiskEstimate'] > threshold]}.items()

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\hospitalMortalityPrediction\\ADMISSIONS_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\hospitalMortalityPrediction.csv"
# target_col = 'HOSPITAL_EXPIRE_FLAG'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\recividismPrediction\\compas-scores-two-years_encoded.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\recividismPrediction.csv"
# target_col = 'is_recid'

# dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\fraudDetection\\transactions.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\fraudDetection.csv"
# target_col = 'isFraud'

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_full_encoded.csv'
# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\e-learning_full_encoded_with_skill.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\e-learning"
# target_col = 'correct'
# categ_cols = ['skill', 'tutor_mode', 'answer_type', 'type']
# user_group_names = ['user_id']
# history_train_fraction = 0.5
# h1_train_size = 300
# h2_train_size = 5000
# h1_epochs = 1000
# h2_epochs = 200
# # diss_weights = range(6)
# # diss_multiply_factor = 0.2  # [0, 1]
# # diss_weights = range(6, 10)
# # diss_multiply_factor = 0.2  # [1.2, 1.8]
# # diss_weights = range(5, 8)
# # diss_multiply_factor = 0.4  # [2, 2.8]
# # diss_weights = range(8, 11)
# # diss_multiply_factor = 0.4  # [3.2, 4]
# diss_weights = range(11, 13)
# diss_multiply_factor = 0.4  # [4.4, 4.8]
# seeds = [0]
# range_stds = range(-30, 30, 2)
# hybrid_stds = list((-x/10 for x in range_stds))
# min_history_size = 200
# max_history_size = 800
# current_user_count = 0
# user_max_count = 32
# split_by_chronological_order = False
# only_hybrid = True
# copy_h1_weights = False

# full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\matific\\matific.csv'
# results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\matific"
# target_col = 'is_finished'
# categ_cols = ['activity_context']
# user_group_names = ['account_id']
# history_train_fraction = 0.5
# h1_train_size = 200
# h2_train_size = 5000
# h1_epochs = 200
# h2_epochs = 200
# diss_weights = range(6)
# diss_multiply_factor = 0.2  # [0, 1]
# seeds = [0]
# range_stds = range(-30, 30, 2)
# hybrid_stds = list((-x/10 for x in range_stds))
# min_history_size = 100
# max_history_size = 800
# current_user_count = 0
# user_max_count = 50
# split_by_chronological_order = False
# only_hybrid = True
# hybrid_method = 'parametric'
# # hybrid_method = 'neural_network'
# copy_h1_weights = False
# df_max_size = 200000
# skip_cols = []
# # skip_cols = ['correct_answers_percentage']

full_dataset_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\mallzee\\mallzee.csv'
results_path = "C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\results\\mallzee"
target_col = 'userResponse'
categ_cols = ['Currency', 'TypeOfClothing', 'Gender', 'InStock', 'Brand', 'Colour']
user_group_names = ['userID']
history_train_fraction = 0.5
h1_train_size = 200
h2_train_size = 5000
h1_epochs = 1000
h2_epochs = 10
layer_1_size = 50
layer_2_size = 10
batch_size = 128

# No hist model
diss_weights = range(5)
diss_multiply_factor = 0.2  # [0, 0.8]

# # L1 model
# diss_weights = range(11)
# diss_multiply_factor = 0.1  # [0, 1]

seeds = [0]
range_stds = range(-30, 30, 2)
hybrid_stds = list((-x/10 for x in range_stds))

# min_history_size = 400
min_history_size = 100
max_history_size = 800
current_user_count = 0
user_max_count = 10

split_by_chronological_order = False
only_hybrid = True
only_L1 = False
hybrid_method = 'parametric'
# hybrid_method = 'neural_network'

copy_h1_weights = False
balance_histories = False
df_max_size = -1
skip_cols = []

plot_confusion = True

# if only_hybrid:
#     diss_weights = [0]

plots_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots'
if os.path.exists(plots_dir):
    shutil.rmtree(plots_dir)
os.makedirs(plots_dir + '\\by_hist_length')
os.makedirs(plots_dir + '\\by_accuracy_range')
os.makedirs(plots_dir + '\\by_compatibility_range')
os.makedirs(plots_dir + '\\by_user_id')

with open(plots_dir + '\\log.csv', 'w', newline='') as file_out:
    writer = csv.writer(file_out)
    header = ['train frac', 'user_id', 'instances', 'train seed', 'comp range', 'acc range', 'h1 acc',
              'baseline x', 'baseline y', 'diss weight', 'no hist x', 'no hist y',
              'L0 x', 'L0 y', 'L1 x', 'L1 y', 'L2 x', 'L2 y']
    writer.writerow(header)

with open(plots_dir + '\\hybrid_log.csv', 'w', newline='') as file_out:
    writer = csv.writer(file_out)
    header = ['train frac', 'user_id', 'instances', 'train seed', 'std offset', 'hybrid x', 'hybrid y']
    writer.writerow(header)

with open(plots_dir + '\\auc.csv', 'w', newline='') as file_out:
    writer = csv.writer(file_out)
    # header = ['train frac', 'user_id', 'instances', 'cos sim', 'train seed', 'comp range', 'acc range', 'h1 acc',
    #           'no hist area', 'hybrid area', 'L0 area', 'L1 area', 'L2 area']
    header = ['train frac', 'user_id', 'instances', 'train seed', 'comp range', 'acc range', 'h1 acc',
              'no hist area', 'hybrid area', 'L0 area', 'L1 area', 'L2 area']
    writer.writerow(header)

print('loading data...')
df_full = pd.read_csv(full_dataset_path)
if df_max_size >= 0:
    df_full = df_full[:df_max_size]

for col in skip_cols:
    try:
        del df_full[col]
    except:
        pass

# one hot encoding
print('pre-processing data... ')
ohe = ce.OneHotEncoder(cols=categ_cols, use_cat_names=True)
df_full = ohe.fit_transform(df_full)
# df_balanced = ohe.transform(df_balanced)

print('splitting into train and test sets...')

# create user groups
user_groups_train = []
user_groups_test = []
for user_group_name in user_group_names:
    user_groups_test += [df_full.groupby([user_group_name])]

# separate histories into training and test sets
students_group = user_groups_test[0]
if split_by_chronological_order:
    df_train = students_group.apply(lambda x: x[:int(len(x) * history_train_fraction) + 1])
else:
    df_train = students_group.apply(lambda x: x.sample(n=int(len(x) * history_train_fraction) + 1, random_state=1))

df_train.index = df_train.index.droplevel(0)
df_test = df_full.drop(df_train.index)
user_groups_test[0] = df_test.groupby([user_group_names[0]])
user_groups_train += [df_train.groupby([user_group_names[0]])]
del df_train[user_group_names[0]]
del df_test[user_group_names[0]]

# balance sets
print('balancing train set...')
target_group = df_train.groupby(target_col)
df_train = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))
df_train = df_train.reset_index(drop=True)
# target_group = df_test.groupby(target_col)
# df_test = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))
# df_test = df_test.reset_index(drop=True)

df_train_subsets_by_seed = []
Xs_by_seed = []
Ys_by_seed = []
h1s_by_seed = []
h2s_not_using_history_by_seed = []

tests_group = {}
tests_group_user_ids = []
for seed in seeds:
    print('SETTING TRAIN SEED '+str(seed+1)+'...')
    df_train_subset = df_train.sample(n=h2_train_size, random_state=seed)
    df_train_subsets_by_seed += [df_train_subset]

    # tests_group[str(seed + 1)] = df_train_subset
    tests_group[str(seed + 1)] = df_test.sample(n=h2_train_size, random_state=seed)
    tests_group_user_ids += [str(seed + 1)]

    X = df_train_subset.loc[:, df_train_subset.columns != target_col]
    Y = df_train_subset[[target_col]]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X, Y)
    labelizer = LabelBinarizer()
    Y = labelizer.fit_transform(Y)

    Xs_by_seed += [X]
    Ys_by_seed += [Y]

    # history_test_x = X[100:].loc[df['ExternalRiskEstimate'] > 75]
    # history_test_y = Y[100:].loc[df['ExternalRiskEstimate'] > 75]

    # min max scale and binarize the target labels
    # X_original = X
    # history_test_x_original = history_test_x

    # history_test_x = scaler.fit_transform(history_test_x, history_test_y)
    # history_test_y = label.fit_transform(history_test_y)

    # h2_without_history_auc = {}
    # new_correct = {}
    # got_without_history = False

    # Dissonance types
    # diss_types = ["D", "D'", "D''"]
    # diss_types = ["D"]

    # user_max_count = 10

    # h1s = []

    # com_range = []
    # acc_range = []
    # h1_train_fractions = range(200, h2_train_fraction, 200)

    # h1_train_fractions = [200]
    # for h1_train_fraction in h1_train_fractions:
    h1 = MLP(X, Y, h1_train_size, h1_epochs, batch_size, layer_1_size, layer_2_size, 0.02, weights_seed=1)
    h1s_by_seed += [h1]

    print("training h2s not using history...")

    h2s_not_using_history = []
    first_diss = True
    for i in diss_weights:
        print('dissonance weight '+str(len(h2s_not_using_history) + 1) + "/" + str(len(diss_weights)))
        diss_weight = diss_multiply_factor * i

        if not first_diss and only_L1:
            h2s_not_using_history += [h2s_not_using_history[0]]
            continue

        h2s_not_using_history += [MLP(X, Y, h2_train_size, h2_epochs, batch_size, layer_1_size, layer_2_size, 0.02, diss_weight, h1, 'D', True,
                                      test_model=False, copy_h1_weights=copy_h1_weights, weights_seed=2)]
        tf.reset_default_graph()
        first_diss = False
    h2s_not_using_history_by_seed += [h2s_not_using_history]

user_group_names.insert(0, 'test')
user_groups_test.insert(0, tests_group)

students_group_user_ids = list(students_group.groups.keys())

# # get users with highest cos_sim between history train and test sets
# students_group_user_ids = []
# users_cos_sim = []
# cos_sim_path = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\DataSets\\e-learning\\analysis\\users\\distributions\\all skills\\train_percent_50\\split similarity\\cos_sim_of_sets.csv'
# with open(cos_sim_path, 'r', newline='') as file_in:
#     reader = csv.reader(file_in)
#     next(reader)
#     for row in reader:
#         students_group_user_ids += [int(row[0])]
#         users_cos_sim += [row[1]]
# # com_range += [abs(h2s[0].compatibility-h2s[1].compatibility)]
# # acc_range += [abs(h2s[0].accuracy-h2s[1].accuracy)]
# # h1_train_fractions = [x / h2_train_fraction for x in h1_train_fractions]
# # plt.plot(h1_train_fractions, com_range, 'b', label='coms', marker='.')
# # plt.plot(h1_train_fractions, acc_range, 'r', label='accs', marker='.')
# # plt.xlabel('h1 train size / h2 train size')
# # # plt.ylabel('')
# # plt.legend(('compatibility range', 'accuracy range'), loc='upper right')
# # plt.title('Varying h1 train sizes, where h2 train size = '+str(h2_train_fraction))
# # plt.savefig('C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots\\different_h1_train_sizes.png')
# # plt.show()
# # break

user_groups_user_ids = [tests_group_user_ids, students_group_user_ids]
user_group_idx = -1

for user_group_test in user_groups_test:
    user_group_idx += 1
    user_group_name = user_group_names[user_group_idx]
    user_ids = user_groups_user_ids[user_group_idx]

    # if user_group_name == 'user_id' or user_group_name == 'teacher_id' or user_group_name == 'student_class_id':
    # if user_group_name == 'teacher_id' or user_group_name == 'student_class_id':
    #     continue


    # for seed in seeds:
    #     plots_dir = 'C:\\Users\\Jonathan\\Documents\\BGU\\Research\\Thesis\\plots\\seed_' + str(seed + 1)
    #     if not os.path.exists(plots_dir):
    #         os.makedirs(plots_dir)
    #     if user_group_name != 'test':
    #         plots_dir += '\\' + user_group_name
    #         if not os.path.exists(plots_dir):
    #             os.makedirs(plots_dir + '\\by_hist_length')
    #             os.makedirs(plots_dir + '\\by_accuracy_range')
    #             os.makedirs(plots_dir + '\\by_compatibility_range')

    total_users = 0
    user_ids_in_range = []
    if user_group_name == 'test':
        total_users = 1
        user_ids_in_range = ['1']
    else:
        for user_id in user_ids:
            try:
                user_test_set = user_group_test.get_group(user_id)
                user_train_set = user_groups_train[user_group_idx - 1].get_group(user_id)

                if balance_histories:
                    target_group = user_test_set.groupby(target_col)
                    user_test_set = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))
                    target_group = user_train_set.groupby(target_col)
                    user_train_set = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))

                history_len = len(user_test_set) + len(user_train_set)
                if min_history_size <= history_len <= max_history_size:
                    total_users += 1
                    user_ids_in_range += [user_id]
            except KeyError:
                pass

    user_count = 0
    # for user_id, user_test_set in user_group_test:
    for user_id in user_ids_in_range:
        if user_group_name == 'test':
            user_test_set = user_group_test[user_id]
        else:
            user_test_set = user_group_test.get_group(user_id)
            if user_count == user_max_count:
                continue

        user_count += 1
        if user_count <= current_user_count:
            continue

        for name in user_group_names:
            try:
                del user_test_set[name]
            except:
                pass

        if balance_histories:
            target_group = user_test_set.groupby(target_col)
            user_test_set = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))

        history_test_x = scaler.transform(user_test_set.loc[:, user_test_set.columns != target_col])
        history_test_y = labelizer.transform(user_test_set[[target_col]])

        if user_group_name != 'test':

            user_train_set = user_groups_train[user_group_idx - 1].get_group(user_id)

            history_len = len(user_test_set) + len(user_train_set)
            # if history_len < min_history_size or history_len > max_history_size:
            #     continue

            for name in user_group_names:
                try:
                    del user_train_set[name]
                except:
                    pass

            if balance_histories:
                target_group = user_train_set.groupby(target_col)
                user_train_set = target_group.apply(lambda x: x.sample(target_group.size().min(), random_state=1))

            history_train_x = scaler.transform(user_train_set.loc[:, user_train_set.columns != target_col])
            history_train_y = labelizer.transform(user_train_set[[target_col]])

            # history = History(history_test_x, history_test_y, 0.01)
            history = History(history_train_x, history_train_y, 0.001)

            # print('training baseline model...')
            if not only_hybrid:
                h2_baseline = MLP(history_train_x, history_train_y, len(user_train_set), h1_epochs, batch_size, layer_1_size, layer_2_size,
                                  weights_seed=2, copy_h1_weights=copy_h1_weights)
        else:
            history_len = len(user_test_set)

        for seed_idx in range(len(seeds)):
            seed = seeds[seed_idx]
            if user_group_name == 'test' and seed_idx != user_count-1:
                continue
            print(
                str(user_count) + '/' + str(total_users) + ' ' + user_group_name + ' ' + str(user_id) + ', instances: ' + str(
                    history_len) + ', seed='+str(seed+1)+'\n')

            confusion_dir = plots_dir + '\\by_user_id\\'+user_group_name+'_'+str(user_id)
            if not os.path.exists(confusion_dir):
                os.makedirs(confusion_dir)

            X = Xs_by_seed[seed_idx]
            Y = Ys_by_seed[seed_idx]
            h1 = h1s_by_seed[seed_idx]
            h2s_not_using_history = h2s_not_using_history_by_seed[seed_idx]

            h1_acc = h1.test(history_test_x, history_test_y)

            plot_confusion_matrix(h1, history_test_x, history_test_y,
                                  user_group_name+'_'+str(user_id)+'h1',
                                  confusion_dir + '\\h1.png', plot_confusion)

            h2_on_history_not_using_history_x = []
            h2_on_history_not_using_history_y = []
            for i in range(len(h2s_not_using_history)):
                h2 = h2s_not_using_history[i]
                result_not_using_history = h2.test(history_test_x, history_test_y, h1)
                h2_on_history_not_using_history_x += [result_not_using_history['compatibility']]
                h2_on_history_not_using_history_y += [result_not_using_history['auc']]

                diss_str = '%.2f' % (diss_weights[i] * diss_multiply_factor)
                plot_confusion_matrix(h2, history_test_x, history_test_y,
                                      user_group_name + '=' + str(user_id) + ' h2=no_hist diss='+diss_str,
                                      confusion_dir + '\\no_hist_'+str(i)+'.png', plot_confusion)

            min_x = min(h2_on_history_not_using_history_x)
            max_x = max(h2_on_history_not_using_history_x)
            min_y = min(h2_on_history_not_using_history_y)
            max_y = max(h2_on_history_not_using_history_y)

            if user_group_name != 'test':
                if not only_hybrid:
                    result_not_using_history = h2_baseline.test(history_test_x, history_test_y, h1)
                    h2_baseline_x = [result_not_using_history['compatibility']]
                    h2_baseline_y = [result_not_using_history['auc']]

                    plot_confusion_matrix(h2_baseline, history_test_x, history_test_y,
                                          user_group_name + '=' + str(user_id) + ' baseline',
                                          confusion_dir + '\\baseline.png', plot_confusion)

                    # h2_baseline_x = ['?']
                    # h2_baseline_y = ['?']

                    # history.set_simple_likelihood(X, h2s_not_using_history[0].W1, magnitude_multiplier=2)
                    if not only_L1:
                        history.set_simple_likelihood(X, magnitude_multiplier=2)
                        # history.set_cheat_likelihood(X_original, threshold)
                    history.set_kernels(X, magnitude_multiplier=10)

                    # min_x = min(min_x, h2_baseline_x[0])
                    # max_x = max(max_x, h2_baseline_x[0])
                    # min_y = min(min_y, h2_baseline_y[0])
                    # max_y = max(max_y, h2_baseline_y[0])

                print('hybrid training...')
                # start_time = int(round(time.time() * 1000))
                # h2s_not_using_history[0].set_hybrid_test(history, history_test_x, history_test_y)
                h2s_not_using_history[0].set_hybrid_test(history, history_test_x, history_test_y, hybrid_method)
                # runtime = str(int((round(time.time() * 1000)) - start_time) / 1000)
                # print("runtime = " + str(runtime) + " secs\n")

                h2_on_history_hybrid_x = []
                h2_on_history_hybrid_y = []

                print('hybrid testing...\n')
                for i in range(len(hybrid_stds)):
                    std = hybrid_stds[i]
                    h2 = h2s_not_using_history[0]  # no dissonance
                    result_hybrid = h2.hybrid_test(history_test_y, std)
                    h2_on_history_hybrid_x += [result_hybrid['compatibility']]
                    h2_on_history_hybrid_y += [result_hybrid['auc']]

                    plot_confusion_matrix(h2, history_test_x, history_test_y,
                                          user_group_name + '=' + str(user_id) + ' h2=hybrid_'+hybrid_method+' std=' + str(std),
                                          confusion_dir + '\\hybrid_'+hybrid_method+'_'+str(i)+'.png', plot_confusion)

                min_x = min(min_x, min(h2_on_history_hybrid_x))
                max_x = max(max_x, max(h2_on_history_hybrid_x))
                min_y = min(min_y, min(h2_on_history_hybrid_y))
                max_y = max(max_y, max(h2_on_history_hybrid_y))

                if not only_hybrid:

                    h2_on_history_L0_x = []
                    h2_on_history_L0_y = []
                    h2_on_history_L1_x = []
                    h2_on_history_L1_y = []
                    h2_on_history_L2_x = []
                    h2_on_history_L2_y = []

                    if only_L1:
                        models = ['L1']
                        h2_on_history_x = [h2_on_history_L1_x]
                        h2_on_history_y = [h2_on_history_L1_y]

                        h2_on_history_L0_x = [1]*len(diss_weights)
                        h2_on_history_L0_y = [1]*len(diss_weights)
                        h2_on_history_L2_x = [1]*len(diss_weights)
                        h2_on_history_L2_y = [1]*len(diss_weights)
                    else:
                        models = ['L0', 'L1', 'L2']
                        h2_on_history_x = [h2_on_history_L0_x, h2_on_history_L1_x, h2_on_history_L2_x]
                        h2_on_history_y = [h2_on_history_L0_y, h2_on_history_L1_y, h2_on_history_L2_y]

                    iteration = 0
                    for i in range(len(diss_weights)):
                        print('dissonance weight '+str(i+1) + "/" + str(len(diss_weights)))
                        diss_weight = diss_weights[i] * diss_multiply_factor
                        for j in range(len(models)):
                            tf.reset_default_graph()
                            model_type = models[j]
                            h2_using_history = MLP(X, Y, h2_train_size, h2_epochs, batch_size, layer_1_size, layer_2_size, 0.02, diss_weight, h1, 'D',
                                                   history=history, use_history=True, model_type=model_type, test_model=False,
                                                   copy_h1_weights=copy_h1_weights, weights_seed=2)
                            result_using_history = h2_using_history.test(history_test_x, history_test_y, h1)
                            h2_on_history_x[j] += [result_using_history['compatibility']]
                            h2_on_history_y[j] += [result_using_history['auc']]

                            diss_str = '%.2f' % (diss_weight)
                            plot_confusion_matrix(h2_using_history, history_test_x, history_test_y,
                                                  user_group_name + '=' + str(user_id) + ' h2='+model_type+' diss=' + diss_str,
                                                  confusion_dir + '\\'+model_type+'_' + str(i) + '.png', plot_confusion)

                    # PLOT
                    # all hist approaches and no hist plot
                    for model in h2_on_history_x:
                        min_model = min(model)
                        if min_x > min_model:
                            min_x = min_model
                        max_model = max(model)
                        if max_x < max_model:
                            max_x = max_model
                    for model in h2_on_history_y:
                        min_model = min(model)
                        if min_y > min_model:
                            min_y = min_model
                        max_model = max(model)
                        if max_y < max_model:
                            max_y = max_model
                        
                get_area = True
                if get_area:
                    mono_h2_on_history_not_using_history_x = h2_on_history_not_using_history_x.copy()
                    mono_h2_on_history_hybrid_x = h2_on_history_hybrid_x.copy()
                    mono_h2_on_history_not_using_history_y = h2_on_history_not_using_history_y.copy()
                    mono_h2_on_history_hybrid_y = h2_on_history_hybrid_y.copy()

                    mono_h2_xs = [mono_h2_on_history_not_using_history_x,
                                  mono_h2_on_history_hybrid_x]
                    mono_h2_ys = [mono_h2_on_history_not_using_history_y,
                                  mono_h2_on_history_hybrid_y]

                    if not only_hybrid:
                        mono_h2_on_history_L0_x = h2_on_history_L0_x.copy()
                        mono_h2_on_history_L1_x = h2_on_history_L1_x.copy()
                        mono_h2_on_history_L2_x = h2_on_history_L2_x.copy()
                        mono_h2_on_history_L0_y = h2_on_history_L0_y.copy()
                        mono_h2_on_history_L1_y = h2_on_history_L1_y.copy()
                        mono_h2_on_history_L2_y = h2_on_history_L2_y.copy()

                        if only_L1:
                            mono_h2_xs += [mono_h2_on_history_L1_x]
                            mono_h2_ys += [mono_h2_on_history_L1_y]
                        else:
                            mono_h2_xs += [mono_h2_on_history_L0_x,
                                           mono_h2_on_history_L1_x,
                                           mono_h2_on_history_L2_x]
                            mono_h2_ys += [mono_h2_on_history_L0_y,
                                           mono_h2_on_history_L1_y,
                                           mono_h2_on_history_L2_y]

                    for i in range(len(mono_h2_xs)):
                        make_monotonic(mono_h2_xs[i], mono_h2_ys[i])

                    h1_area = (1-min_x)*h1_acc
                    h2_on_history_not_using_history_area = auc([min_x] + mono_h2_on_history_not_using_history_x + [1],
                                                               [mono_h2_on_history_not_using_history_y[0]] + mono_h2_on_history_not_using_history_y + [h1_acc]) - h1_area
                    h2_on_history_hybrid_area = auc([min_x] + mono_h2_on_history_hybrid_x + [1],
                                                    [mono_h2_on_history_hybrid_y[0]] + mono_h2_on_history_hybrid_y + [h1_acc]) - h1_area

                    if not only_hybrid:
                        h2_on_history_L1_area = auc([min_x] + mono_h2_on_history_L1_x + [1],
                                                    [mono_h2_on_history_L1_y[0]] + mono_h2_on_history_L1_y + [h1_acc]) - h1_area
                        h2_on_history_L0_area = auc([min_x] + mono_h2_on_history_L0_x + [1],
                                                    [mono_h2_on_history_L0_y[0]] + mono_h2_on_history_L0_y + [h1_acc]) - h1_area
                        h2_on_history_L2_area = auc([min_x] + mono_h2_on_history_L2_x + [1],
                                                    [mono_h2_on_history_L2_y[0]] + mono_h2_on_history_L2_y + [h1_acc]) - h1_area

                com_range = max_x - min_x
                auc_range = max_y - min_y

                h1_x = [min_x, max_x]
                h1_y = [h1_acc, h1_acc]
                plt.plot(h1_x, h1_y, 'k--', marker='.', label='h1')

                if not only_hybrid:
                    # plt.plot(h2_baseline_x, h2_baseline_y, 'ks', markersize=10, label='h2 baseline')
                    plt.plot(h2_on_history_not_using_history_x, h2_on_history_not_using_history_y, 'b', marker='.', linewidth=6, markersize=22, label='h2 not using history')
                    plt.plot(h2_on_history_hybrid_x, h2_on_history_hybrid_y, 'g', marker='.', linewidth=5, markersize=18, label='h2 hybrid')
                    if not only_L1:
                        plt.plot(h2_on_history_L0_x, h2_on_history_L0_y, 'r', marker='.', linewidth=4, markersize=14, label='h2 using L0')
                        plt.plot(h2_on_history_L2_x, h2_on_history_L2_y, 'orange', marker='.', linewidth=2, markersize=6, label='h2 using L2')
                    plt.plot(h2_on_history_L1_x, h2_on_history_L1_y, 'm', marker='.', linewidth=3, markersize=10, label='h2 using L1')
                else:
                    plt.plot(h2_on_history_not_using_history_x, h2_on_history_not_using_history_y, 'b', marker='.', linewidth=4, markersize=14, label='h2 not using history')
                    plt.plot(h2_on_history_hybrid_x, h2_on_history_hybrid_y, 'g', marker='.', linewidth=3, markersize=10, label='h2 hybrid')

                plt.xlabel('compatibility')
                plt.ylabel('accuracy')
                plt.grid()
                plt.legend()

                # plt.title(
                #     'user=' + str(user_id) + ' hist_len=' + str(history_len) + ' split=' + str(
                #         100 * history_train_fraction) + '% sim=' + str(users_cos_sim[user_idx])+' seed=' + str(seed + 1))

                plt.title(
                    'user=' + str(user_id) + ' hist_len=' + str(history_len) + ' split=' + str(
                        100 * history_train_fraction) + ' seed=' + str(
                        seed + 1))

                plt.savefig(
                    plots_dir + '\\by_hist_length\\len_' + str(history_len) + '_' + user_group_name + '_' + str(
                        user_id) + '.png')
                plt.savefig(
                    plots_dir + '\\by_accuracy_range\\acc_' + '%.4f' % (auc_range,) + '_' + user_group_name + '_' + str(
                        user_id) + '.png')
                plt.savefig(
                    plots_dir + '\\by_compatibility_range\\com_' + '%.4f' % (com_range,) + '_' + user_group_name + '_' + str(
                        user_id) + '.png')
                plt.savefig(
                    plots_dir + '\\by_user_id\\'+user_group_name+'_'+str(user_id) + '.png')
                    # plots_dir + '\\by_user_id\\'+user_group_name+'_'+str(user_id) + '_seed_'+str(seed+1)+'.png')

                plt.savefig(confusion_dir + '\\plot.png')
                # plt.show()

                with open(plots_dir + '\\log.csv', 'a', newline='') as file_out:
                    writer = csv.writer(file_out)
                    for i in range(len(diss_weights)):
                        if not only_hybrid:
                            row = [
                                str(history_train_fraction),
                                str(user_id),
                                str(history_len),
                                str(seed + 1),
                                str(com_range),
                                str(auc_range),
                                str(h1_acc),
                                str(h2_baseline_x[0]),
                                str(h2_baseline_y[0]),
                                str(diss_weights[i]*diss_multiply_factor),
                                str(h2_on_history_not_using_history_x[i]),
                                str(h2_on_history_not_using_history_y[i]),
                                str(h2_on_history_L0_x[i]),
                                str(h2_on_history_L0_y[i]),
                                str(h2_on_history_L1_x[i]),
                                str(h2_on_history_L1_y[i]),
                                str(h2_on_history_L2_x[i]),
                                str(h2_on_history_L2_y[i])
                            ]
                        else:
                            row = [
                                str(history_train_fraction),
                                str(user_id),
                                str(history_len),
                                str(seed + 1),
                                str(com_range),
                                str(auc_range),
                                str(h1_acc),
                                '0',
                                '0',
                                str(diss_weights[i] * diss_multiply_factor),
                                str(h2_on_history_not_using_history_x[i]),
                                str(h2_on_history_not_using_history_y[i]),
                                str(h2_on_history_not_using_history_x[i]),
                                str(h2_on_history_not_using_history_y[i]),
                                str(h2_on_history_not_using_history_x[i]),
                                str(h2_on_history_not_using_history_y[i]),
                                str(h2_on_history_not_using_history_x[i]),
                                str(h2_on_history_not_using_history_y[i])
                            ]
                        writer.writerow(row)

                with open(plots_dir + '\\hybrid_log.csv', 'a', newline='') as file_out:
                    writer = csv.writer(file_out)
                    for i in range(len(hybrid_stds)):
                        row = [
                            str(history_train_fraction),
                            str(user_id),
                            str(history_len),
                            # str(users_cos_sim[user_idx]),
                            str(seed + 1),
                            str(hybrid_stds[i]),
                            str(h2_on_history_hybrid_x[i]),
                            str(h2_on_history_hybrid_y[i])
                        ]
                        writer.writerow(row)

                with open(plots_dir + '\\auc.csv', 'a', newline='') as file_out:
                    writer = csv.writer(file_out)
                    row = [
                        str(history_train_fraction),
                        str(user_id),
                        str(history_len),
                        # str(users_cos_sim[user_idx]),
                        str(seed + 1),
                        str(com_range),
                        str(auc_range),
                        str(h1_acc),
                        str(h2_on_history_not_using_history_area),
                        str(h2_on_history_hybrid_area)
                    ]
                    if not only_hybrid:
                        row += [
                            str(h2_on_history_L0_area),
                            str(h2_on_history_L1_area),
                            str(h2_on_history_L2_area)
                        ]
                    writer.writerow(row)

            else:  # on test
                # if not only_hybrid:
                h1_x = [min_x, max_x]
                h1_y = [h1_acc, h1_acc]
                plt.plot(h1_x, h1_y, 'k--', marker='.', label='h1')
                plt.plot(h2_on_history_not_using_history_x, h2_on_history_not_using_history_y, 'b', marker='.', label='h2')
                plt.xlabel('compatibility')
                plt.ylabel('accuracy')
                plt.legend(loc='lower left')
                plt.title('test seed=' + str(user_id) + ' h1=' + str(h1_train_size) + ' h2=' + str(
                    h2_train_size)+' len='+str(history_len))

                plt.savefig(plots_dir+'\\test_for_seed_' + str(user_id) + '.png')
                plt.savefig(confusion_dir+'\\plot.png')
                plt.show()

            plt.clf()
