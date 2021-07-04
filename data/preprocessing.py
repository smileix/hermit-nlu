import numpy as np
from collections import OrderedDict


def reshape_for_training(dataset):
    num_input = len(dataset[0][0])
    num_label = len(dataset[0][1])
    X = []
    Y = []
    for i in range(num_input):
        X.append([])
    for i in range(num_label):
        Y.append([])
    for example in dataset:
        for i in range(len(example[0])):
            X[i].append(example[0][i])
        for i in range(len(example[1])):
            Y[i].append(example[1][i])
    return X, Y


def unpad(gold, prediction, label_encoders=None):
    y = OrderedDict()
    for i, label in enumerate(label_encoders):
        gold_standard = np.argmax(gold[i], axis=2)
        predicted = np.argmax(prediction[i], axis=2)
        y_true = []
        y_pred = []
        for y_true_i, y_pred_i in zip(gold_standard, predicted):
            new_true = []
            new_pred = []
            for true_token, pred_token in zip(y_true_i, y_pred_i):
                if true_token != 0:
                    new_true.append(true_token)
                    new_pred.append(pred_token)
            y_true.append(label_encoders[label].inverse_transform(new_true).tolist())
            y_pred.append(label_encoders[label].inverse_transform(new_pred).tolist())
            del new_true
            del new_pred
        y[label] = (y_true, y_pred)
        del y_true
        del y_pred
        del gold_standard
        del predicted
    return y
