# -*- coding: UTF-8 -*-
import argparse
import os
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description='HERMIT NLU - HiERarchical MultI-Task Natural Language Understanding')
parser.add_argument('-n', '--network', type=str, default='hermit',
                    help='Network topology [default="hermit"]')
parser.add_argument('-d', '--dataset', type=str, help='Dataset to load. The dataset will be split into K-folds.')
parser.add_argument('--train-set', type=str, default=None,
                    help='Training set. Only for static split (requires test set).')
parser.add_argument('--test-set', type=str, default=None,
                    help='Test set. Only for static split (requires train set).')
parser.add_argument('-m', '--mode', type=str, default='testing', help='Mode [training|testing]')
parser.add_argument('--units', nargs='+', type=int, default=[150, 200, 250], help='Units (LSTM)')
parser.add_argument('--dropout', nargs='+', type=float, default=[0.2, 0.8], help='Dropout rates')
parser.add_argument('--batch-size', nargs='+', type=int, default=[32], help='Batch size')
parser.add_argument('--loss', nargs='+', type=str, default=['categorical_crossentropy'], help='Losses')
parser.add_argument('--optimizer', nargs='+', type=str, default=['adam', 'rmsprop'],
                    help='Optimizer')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs (default=300)')
parser.add_argument('--patience', type=int, default=20, help='Patience for Early Stopping (default=20, -1 to disable)')
parser.add_argument('-g', '--gpu', type=str, default='', help='GPU ids, comma separated [e.g., 0,1]')
parser.add_argument('--num-threads', type=int, default=0, help='Number of threads')
parser.add_argument('--run-folder', type=str, help='Run output folder')
parser.add_argument('--monitor', type=str, nargs='+', default=['f1'], help='Monitoring measure for early stopping')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import learning.network as net
import data.dataset as data


hyper_params = dict()
hyper_params['units'] = args.units
hyper_params['dropout'] = args.dropout
hyper_params['batch_size'] = args.batch_size
hyper_params['loss'] = args.loss
hyper_params['optimizer'] = args.optimizer
hyper_params['monitor_m'] = args.monitor

print('Running in {} mode'.format(args.mode))

network = args.network

examples = []
train_set = []
test_set = []

if args.train_set is not None and args.test_set is not None:
    d = data.Dataset(train_set_path=args.train_set, test_set_path=args.test_set)
    train_set, test_set, label_encoders = d.generate_training_data(feature_spaces=[],
                                                                   run_folder=args.run_folder)
else:
    d = data.Dataset(dataset_path=args.dataset)
    examples, label_encoders = d.generate_training_data(feature_spaces=[],
                                                        run_folder=args.run_folder)
hyper_params['attention_activation'] = ['tanh']
hyper_params['attention_width'] = [d.max_sentence_length]

hyper_params = ParameterGrid(hyper_params)

net = net.KerasNetwork(network=args.network,
                       label_encoders=label_encoders,
                       embedding_matrices=d.embeddings,
                       num_threads=args.num_threads)

file_name = args.network
if args.mode == 'training':
    net.train(examples=examples, epochs=args.epochs, patience=args.patience, hyper_params=hyper_params)
    net.save_model(name=file_name, run_folder=args.run_folder)
elif args.mode == 'testing':
    predictions_folder = os.path.join('resources', args.run_folder, 'predictions')

    if args.train_set is not None and args.test_set is not None:
        fold_name = os.path.basename(os.path.dirname(args.train_set))
        file_name = file_name + '_' + fold_name
        results = net.evaluate_static_split(train_examples=train_set,
                                            test_examples=test_set,
                                            epochs=args.epochs,
                                            patience=args.patience,
                                            hyper_params=hyper_params,
                                            predictions_folder=predictions_folder,
                                            fold_name=fold_name,
                                            val_percentage=0.1)
    else:
        results = net.evaluate(examples=examples,
                               epochs=args.epochs,
                               patience=args.patience,
                               hyper_params=hyper_params,
                               predictions_folder=predictions_folder,
                               n_folds=10)
    out_file = os.path.join('resources', args.run_folder, 'results', file_name + '.txt')
    net.print_evaluation_report(out_file)
