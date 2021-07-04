from loaders import SSPLoader
import argparse
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import sys
import os
from shutil import copyfile
import ntpath
import math

NO_FRAME = "[NO_FRAME]"


class FrameSplitter:
    """

    """
    def __init__(self, dataset, balanced=False):
        self.enc = MultiLabelBinarizer()
        self.binary = self.enc.fit_transform(dataset)
        self.balanced = balanced

    def nfold(self, n_folds, devel=None):
        return FrameSplitter.deal_with_nfold(self.binary,
                                             n_folds,
                                             devel=devel,
                                             balanced=self.balanced)

    def fixed_split(self, train=0.8, devel=0, test=0.2):
        return FrameSplitter.deal_with_fixed(self.binary,
                                             train=train,
                                             devel=devel,
                                             test=test,
                                             balanced=self.balanced)

    @staticmethod
    def deal_with_nfold(binary_repr, n_folds, devel=None, balanced=False):
        if balanced:
            folders = prob_mass_nfold(binary_repr, n_folds)
        else:
            folders = nfold_split(binary_repr, n_folds)

        n_folding = []

        for i, folder in enumerate(folders):
            test = folder
            train = folders[:i] + folders[i + 1:]
            dev = []
            if devel:
                mode = devel[0]
                param = devel[1]
                if mode == "folds":
                    dev = train[:param]
                    dev = [idx for sublist in dev for idx in sublist]
                    train = train[param:]
                    train = [idx for sublist in train for idx in sublist]

                elif mode == "fixed":
                    # this is needed to put together all the binary repr of
                    # the current train, and use them in prob_mass_split
                    this_train_binary = [binary_repr[idx] for sublist in train for idx in sublist]
                    if balanced:
                        train, dev = prob_mass_split(this_train_binary,
                                                             1 - param)  # because devel[1] contains the devel perc
                    else:
                        print("non balanced not implemented yet")
            else:
                train = [idx for sublist in train for idx in sublist]

            folding = {"train": train, "devel": dev, "test": test}
            n_folding.append(folding)

        return n_folding

    @staticmethod
    def deal_with_fixed(binary_repr, train=0.8, devel=0, test=0.2, balanced=False):
        if train > 1 or test > 1 or devel > 1:
            print("Warning: some splitting values are higher than 1")
            return None
        if train + test > 1:
            print("Warning: train+test must sum up to 1")
            return None

        if balanced:
            train, test = prob_mass_split(binary_repr, train)
        else:
            train, test = fixed_split(binary_repr, train)

        if devel > 0:
            train_binary = [binary_repr[idx] for idx in train]
            if balanced:
                train, dev = prob_mass_split(train_binary, 1 - devel)
            else:
                train, dev = fixed_split(binary_repr, train)
        else:
            dev = []

        return {"train": train, "devel": dev, "test": test}

    @staticmethod
    def save_splitting(splitting, dataset, ouput):
        os.makedirs(ouput, exist_ok=False)

        if type(splitting) == list:
            for i, fold in enumerate(splitting):
                fold_idx = str(i)
                os.makedirs(os.path.join(ouput, fold_idx, "train"), exist_ok=True)
                os.makedirs(os.path.join(ouput, fold_idx, "test"), exist_ok=True)

                for idx in fold["train"]:
                    file_path = dataset[idx]["name"]
                    copyfile(file_path, os.path.join(ouput, fold_idx, "train", ntpath.basename(file_path)))
                if len(fold["devel"]) > 0:
                    os.makedirs(os.path.join(ouput, fold_idx, "devel"), exist_ok=True)
                    for idx in fold["devel"]:
                        file_path = dataset[idx]["name"]
                        copyfile(file_path, os.path.join(ouput, fold_idx, "devel", ntpath.basename(file_path)))
                for idx in fold["test"]:
                    file_path = dataset[idx]["name"]
                    copyfile(file_path, os.path.join(ouput, fold_idx, "test", ntpath.basename(file_path)))

                print("Fold", i)
                print("Train len={}".format(len(fold["train"])))
                print("Devel len={}".format(len(fold["devel"])))
                print("Test len={}".format(len(fold["test"])))
        elif type(splitting) == dict:
            os.makedirs(os.path.join(ouput, "train"), exist_ok=True)
            os.makedirs(os.path.join(ouput, "test"), exist_ok=True)

            for idx in splitting["train"]:
                file_path = dataset[idx]["name"]
                copyfile(file_path, os.path.join(ouput, "train", ntpath.basename(file_path)))
            if len(splitting["devel"]) > 0:
                os.makedirs(os.path.join(ouput, "devel"), exist_ok=True)
                for idx in splitting["devel"]:
                    file_path = dataset[idx]["name"]
                    copyfile(file_path, os.path.join(ouput, "devel", ntpath.basename(file_path)))
            for idx in splitting["test"]:
                file_path = dataset[idx]["name"]
                copyfile(file_path, os.path.join(ouput, "test", ntpath.basename(file_path)))

            print("Train len={}".format(len(splitting["train"])))
            print("Devel len={}".format(len(splitting["devel"])))
            print("Test len={}".format(len(splitting["test"])))


def init_argparse():
    parser = argparse.ArgumentParser(
        description='Data preparation script: organise a hrc3 dataset in train/test/devel/nfold')
    parser.add_argument("input",
                        help="directory containing the diafram hrc3 corpus")
    parser.add_argument("-d", "--dest",
                        type=str,
                        help="destination directory for new generated experiment split",
                        default=".")
    parser.add_argument("-n", "--nfold",
                        help="n-fold splitting, requires number of fold",
                        type=int)
    parser.add_argument("-f", "--fixed",
                        nargs='+',
                        help="fixed splitting percentages, "
                             "expects either [train_p, test_p] or [train_p, devel_p, test_p] (as floats)",
                        type=float)
    return parser


def get_frames_per_sentence_list(dataset):
    sentences_frames = []

    for sentence in dataset:
        sentence_frames = []
        if len(sentence["frame_semantics"]) == 0:
            sentence_frames.append(NO_FRAME)
        else:
            for frame in sentence["frame_semantics"]:
                sentence_frames.append(frame["frame"])
        sentences_frames.append(sentence_frames)

    return sentences_frames


def nfold_split(y, folds=4):
    n_folds = [[] for x in range(folds)]
    for i, exp in enumerate(y):
        n_folds[i % folds].append(i)
    return n_folds


def prob_mass_nfold(y, folds=4):
    """ performs a stratified nfold split for multilabel classification """
    y = np.array(y)
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    return index_list


def fixed_split(y, train=0.8):
    train_idx = math.ceil(len(y)*train)
    train = [i for i in range(0, train_idx)]
    test = [i for i in range(train_idx, len(y))]
    return train, test


def prob_mass_split(y, train=0.8):
    """ performs a stratified split for multilabel classification """
    decimal_split = 100
    index_list = prob_mass_nfold(y, folds=decimal_split)
    train_perc = int(train*100)
    train_lists = index_list[:train_perc]
    test_lists = index_list[train_perc:]
    train_set = [idx for sublist in train_lists for idx in sublist]
    test_set = [idx for sublist in test_lists for idx in sublist]
    return train_set, test_set


def nfold_dev_selection():
    split = input("Do you want to preserve a devel section in your training folders? [y/N] ")
    if split.lower() == "y":
        mode = input("Do you want the devel section to be a number of folders or a fixed split? [folds/fixed] ")
        if mode == "folds":
            n_devel_folds = input("Please enter the number of folders to use as devel set? ")
            return mode, int(n_devel_folds)
        elif mode == "fixed":
            perc = input("Please enter the percentage (as float) to be reserved as devel set? ")
            return mode, float(perc)
        else:
            print("Wrong value")
            sys.exit(-1)
    elif split.upper() == "N":
        return None
    else:
        print("{} is not a possible answer".format(split))
        sys.exit(-1)


if __name__ == "__main__":
    arg_parser = init_argparse()
    args = arg_parser.parse_args()

    if args.nfold and args.fixed:
        print("Can't perform both a n-fold split and a fixed split")
        sys.exit(-1)

    if args.nfold:
        dev = nfold_dev_selection()

    input_dir = args.input
    loader = SSPLoader()
    loader.load_diafram(input_dir, ann_type='hrc2')
    print(len(loader.dataset))
    frames = get_frames_per_sentence_list(loader.dataset)
    fs = FrameSplitter(frames, balanced=True)
    print("Dataset len={}".format(len(loader.dataset)))

    if args.nfold:
        n_folding = fs.nfold(args.nfold, devel=dev)
        fs.save_splitting(n_folding, loader.dataset, args.dest)
    elif args.fixed:
        if len(args.fixed) > 3 or len(args.fixed) < 1:
            print("wrong number of fixed split parameters")
            sys.exit(-1)
        else:
            params = args.fixed
            if len(params) == 1:
                if params[0] > 1:
                    print("Warning: some splitting values are higher than 1")
                print("Warning: single value passed as fixed split percentage. "
                      "Implied as train set percentage. 1-{} used as test set percentage. ".format(params[0]))
                split = fs.fixed_split(train=params[0], test=1-params[0])
            elif len(params) == 2:
                if params[0] > 1 or params[1] > 1:
                    print("Warning: some splitting values are higher than 1")
                if params[0] + params[1] > 1:
                    print("Warning: train+test must sum up to 1")
                split = fs.fixed_split(train=params[0], test=params[1])
            elif len(params) == 3:
                if params[0] > 1 or params[1] > 1 or params[2] > 1:
                    print("Warning: some splitting values are higher than 1")
                if params[0] + params[2] > 1:
                    print("Warning: train+test must sum up to 1")
                split = fs.fixed_split(train=params[0], devel=params[1], test=params[2])
            fs.save_splitting(split, loader.dataset, args.dest)
