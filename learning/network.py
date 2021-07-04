# -*- coding: UTF-8 -*-
import os
from keras import backend as k
import tensorflow as tf
from keras.models import model_from_json
import numpy as np
import architectures
import inspect
import json

from learning.metrics.sequence_labeling import classification_report
from data.preprocessing import reshape_for_training, unpad
from collections import OrderedDict
from layers.embeddings import ElmoEmbedding
from keras_self_attention import SeqSelfAttention
from learning.layers.crf import CRF


np.random.seed(42)
tf.set_random_seed(42)


class KerasNetwork(object):

    def __init__(self, network, label_encoders, embedding_matrices=None, num_threads=0):
        self.network = network
        self.label_encoders = label_encoders
        self.embedding_matrices = embedding_matrices
        self.num_threads = num_threads
        self.labels = OrderedDict()
        for label in label_encoders:
            self.labels[label] = len(label_encoders[label].classes_)

        available_networks = inspect.getmembers(architectures, inspect.isfunction)
        self.available_networks = {x[0]: x[1] for x in available_networks}
        self.model = None
        self.evaluation_report = OrderedDict()
        self.tuning_report = OrderedDict()
        print('Network initialized!')

    def create_network(self, hyper_params):
        if k.backend() == 'tensorflow':
            k.clear_session()
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=self.num_threads,
                                          inter_op_parallelism_threads=self.num_threads)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            k.set_session(sess)
        self.model = self.available_networks[self.network](hyper_params=hyper_params,
                                                           embedding_matrices=self.embedding_matrices,
                                                           labels=self.labels)

    def tune(self, train_set, val_set, epochs, patience, hyper_params, fold=1, n_folds=1):
        print('\n=== Tuning network ===')
        train_x, train_y = reshape_for_training(train_set)
        val_x, val_y = reshape_for_training(val_set)
        del train_set
        del val_set
        best_f1 = 0
        best_hyper_params = None
        best_epoch = 0
        for setting_counter, current_hyper_params in enumerate(hyper_params):
            print('\nCurrent hyper-params: {}'.format(current_hyper_params))
            rounds_without_improvements = 0
            local_best_f1 = 0
            self.create_network(current_hyper_params)
            for i in range(epochs):
                print('Epoch {}/{} (Patience {}/{}) - Setting {}/{} - Fold {}/{}'.format(i + 1,
                                                                                         epochs,
                                                                                         rounds_without_improvements,
                                                                                         patience,
                                                                                         setting_counter + 1,
                                                                                         len(hyper_params),
                                                                                         fold,
                                                                                         n_folds))

                self.model.fit(x=train_x, y=train_y,
                               batch_size=current_hyper_params['batch_size'],
                               epochs=1,
                               shuffle=True,
                               verbose=1)
                prediction = self.model.predict(val_x, batch_size=current_hyper_params['batch_size'], verbose=1)

                y = unpad(val_y, prediction, self.label_encoders)
                del prediction
                report = classification_report(y)
                del y
                total_f1 = report[current_hyper_params['monitor_m']]['total']

                if total_f1 >= best_f1:
                    best_hyper_params = current_hyper_params
                    best_f1 = total_f1
                    best_epoch = i
                if total_f1 >= local_best_f1:
                    local_best_f1 = total_f1
                    rounds_without_improvements = 0
                    for metric in report:
                        for label in report[metric]:
                            print(' - {} {}: {}'.format(metric, label, report[metric][label])),
                        print('')
                else:
                    rounds_without_improvements += 1
                if rounds_without_improvements == patience:
                    break
                del report
            del local_best_f1
            del rounds_without_improvements
            del self.model
        del train_x
        del train_y
        del val_x
        del val_y
        print('Best F1: {}'.format(best_f1))
        print('Best epoch: {}'.format(best_epoch + 1))
        print('Best hyper-params: {}'.format(best_hyper_params))
        return best_hyper_params, best_epoch + 1

    def evaluate(self, examples, epochs, patience, hyper_params, predictions_folder=None, n_folds=10):
        self.evaluation_report = OrderedDict()
        self.tuning_report = OrderedDict()
        for hyper_param in list(hyper_params)[0]:
            self.tuning_report[hyper_param] = []
        np.random.shuffle(examples)
        folds = [examples[i::n_folds] for i in range(n_folds)]
        for test_i in range(n_folds):
            train_set = []
            val_i = np.random.randint(0, n_folds - 1)
            while val_i == test_i:
                val_i = np.random.randint(0, n_folds - 1)
            for train_i in range(n_folds):
                if train_i != test_i and train_i != val_i:
                    train_set += folds[train_i]
            best_hyper_params, best_epoch = self.tune(train_set=train_set, val_set=folds[val_i],
                                                      epochs=epochs, patience=patience,
                                                      hyper_params=hyper_params, fold=test_i + 1, n_folds=n_folds)
            for best_hyper_param in best_hyper_params:
                self.tuning_report[best_hyper_param].append(best_hyper_params[best_hyper_param])
            print('\n=== Training network ===')
            train_set += folds[val_i]
            self.create_network(best_hyper_params)
            train_x, train_y = reshape_for_training(train_set)
            del train_set
            self.model.fit(x=train_x, y=train_y,
                           batch_size=best_hyper_params['batch_size'],
                           epochs=best_epoch,
                           shuffle=True,
                           verbose=1)
            del train_x
            del train_y
            test_x, test_y = reshape_for_training(folds[test_i])
            prediction = self.model.predict(test_x, batch_size=best_hyper_params['batch_size'], verbose=1)
            y = unpad(test_y, prediction, self.label_encoders)
            report = classification_report(y)
            predictions_file = os.path.join(predictions_folder,
                                            self.network + "_predictions_fold_" + str(test_i + 1) + ".json")
            self.print_predictions(x=test_x, y=y, predictions_file=predictions_file)
            del test_x
            del test_y
            del prediction
            for metrics in report:
                try:
                    self.evaluation_report[metrics]
                except KeyError:
                    self.evaluation_report[metrics] = OrderedDict()
                for label in report[metrics]:
                    try:
                        self.evaluation_report[metrics][label]
                    except KeyError:
                        self.evaluation_report[metrics][label] = []
            for metrics in report:
                for label in report[metrics]:
                    self.evaluation_report[metrics][label].append(report[metrics][label])
            self.print_evaluation_report(out_file='/tmp/partial_results.log')
            del report
            del best_hyper_params
            del best_epoch
            del self.model

    def evaluate_static_split(self, train_examples, test_examples, epochs, patience, hyper_params,
                              predictions_folder=None, fold_name=None, val_percentage=0.1):
        self.evaluation_report = OrderedDict()
        self.tuning_report = OrderedDict()
        for hyper_param in list(hyper_params)[0]:
            self.tuning_report[hyper_param] = []
        self.tuning_report['epoch'] = []
        np.random.shuffle(train_examples)
        np.random.shuffle(test_examples)
        examples_in_validation = int(round(val_percentage * len(train_examples)))
        train_set = train_examples[examples_in_validation:]
        val_set = train_examples[:examples_in_validation]
        best_hyper_params, best_epoch = self.tune(train_set=train_set, val_set=val_set,
                                                  epochs=epochs, patience=patience,
                                                  hyper_params=hyper_params)
        for best_hyper_param in best_hyper_params:
            self.tuning_report[best_hyper_param].append(best_hyper_params[best_hyper_param])
        self.tuning_report['epoch'].append(best_epoch)
        print('\n=== Training network ===')
        train_set += val_set
        np.random.shuffle(train_set)
        self.create_network(best_hyper_params)
        train_x, train_y = reshape_for_training(train_set)
        del train_set
        self.model.fit(x=train_x, y=train_y,
                       batch_size=best_hyper_params['batch_size'],
                       epochs=best_epoch,
                       shuffle=True,
                       verbose=1)
        del train_x
        del train_y
        test_x, test_y = reshape_for_training(test_examples)
        prediction = self.model.predict(test_x, batch_size=best_hyper_params['batch_size'], verbose=1)
        y = unpad(test_y, prediction, self.label_encoders)
        report = classification_report(y)
        predictions_file = os.path.join(predictions_folder, self.network + "_predictions_fold_" + fold_name + ".json")
        self.print_predictions(x=test_x, y=y, predictions_file=predictions_file)
        del test_x
        del test_y
        del prediction
        for metrics in report:
            try:
                self.evaluation_report[metrics]
            except KeyError:
                self.evaluation_report[metrics] = OrderedDict()
            for label in report[metrics]:
                try:
                    self.evaluation_report[metrics][label]
                except KeyError:
                    self.evaluation_report[metrics][label] = []
        for metrics in report:
            for label in report[metrics]:
                self.evaluation_report[metrics][label].append(report[metrics][label])
        self.print_evaluation_report(out_file='/tmp/partial_results.log')
        del report
        del best_hyper_params
        del best_epoch
        del self.model

    def train(self, examples, epochs, patience, hyper_params, n_splits=10):
        np.random.shuffle(examples)
        folds = [examples[i::n_splits] for i in range(n_splits)]
        train_set = []
        val_i = np.random.randint(0, n_splits - 1)
        for train_i in range(n_splits):
            if train_i != val_i:
                train_set += folds[train_i]
        best_hyper_params, best_epoch = self.tune(train_set=train_set, val_set=folds[val_i],
                                                  epochs=epochs, patience=patience,
                                                  hyper_params=hyper_params)
        print('\n=== Training model ===')
        self.create_network(best_hyper_params)
        train_x, train_y = reshape_for_training(examples)
        self.model.fit(x=train_x, y=train_y, batch_size=best_hyper_params['batch_size'],
                       epochs=best_epoch, shuffle=True, verbose=1)
        del train_x
        del train_y
        del train_set

    def save_model(self, name, run_folder):
        models_path = os.path.join('resources', run_folder, 'models', name)
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        print('Saving ' + name + ' model...'),
        open(os.path.join(models_path, 'architecture.json'), 'w').write(self.model.to_json())
        self.model.save_weights(os.path.join(models_path, 'weights.h5'), overwrite=True)
        print('done!')

    def load_model(self, architecture_path, weights_path):
        print('Loading model...'),
        json_file = open(architecture_path, 'r')
        architecture_json = json_file.read()
        json_file.close()
        self.model = model_from_json(architecture_json, custom_objects={'ElmoEmbedding': ElmoEmbedding,
                                                                        'SeqSelfAttention': SeqSelfAttention,
                                                                        'CRF': CRF})
        self.model.load_weights(weights_path)
        print('done!')

    def print_evaluation_report(self, out_file='results.txt'):
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        with open(out_file, 'w') as f:
            for hyper_param in self.tuning_report:
                print(hyper_param + ': ' + str(self.tuning_report[hyper_param]))
                f.write(hyper_param + ': ' + str(self.tuning_report[hyper_param]) + '\n')
            for metrics in self.evaluation_report:
                print(metrics + ':')
                f.write(metrics + ':\n')
                for label in self.evaluation_report[metrics]:
                    print('\t' + label + ': ' + str(self.evaluation_report[metrics][label]))
                    f.write('\t' + label + ': ' + str(self.evaluation_report[metrics][label]) + '\n')
                print('')
                f.write('\n')

    @staticmethod
    def print_predictions(x, y, predictions_file='predictions.txt'):
        if not os.path.exists(os.path.dirname(predictions_file)):
            os.makedirs(os.path.dirname(predictions_file))

        x = map(lambda ex: [token for token in ex if token != '__PAD__'], x[0])

        json_array = []

        for i, tokens in enumerate(x):
            example = OrderedDict()
            example["tokens"] = tokens
            for label, annotations in y.items():
                gold, pred = annotations
                example[label + "_gold"] = gold[i]
                example[label + "_pred"] = pred[i]
            json_array.append(example)

        with open(predictions_file, 'w') as f:
            json.dump(json_array, f, indent=2)
