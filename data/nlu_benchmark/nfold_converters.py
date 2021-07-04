import converters
from os import listdir, makedirs
from os.path import isfile, join, isdir
import argparse


def init_argparse():
    parser = argparse.ArgumentParser(
        description='Convert all the Kfolds of the NLU-benchmark in hrc2 format')
    parser.add_argument("input",
                        help="the CrossValidation directory in the NLU-benchmark")
    parser.add_argument("-o", "--out",
                        type=str,
                        help="destination directory where to save the KFold with the hrc2 files",
                        default=".")
    return parser


def convert_nfold(nfold_input, nfold_output):
    folds = listdir(nfold_input)
    folds = [fold for fold in folds if isdir(join(nfold_input, fold))]

    for fold in folds:
        print(fold)
        fold_fullpath = join(nfold_input, fold)

        # generate train set
        train_dir = join(fold_fullpath, "trainset")
        train_files = [join(train_dir, f) for f in listdir(train_dir)
                       if isfile(join(train_dir, f)) and not f.startswith(".")]
        train_out_dir = join(nfold_output, fold, "trainset")
        makedirs(train_out_dir)
        for train_file in train_files:
            converters.nlub_to_hrc2(train_file, train_out_dir, verbose=False)

        # generate test set
        test_dir = join(fold_fullpath, "testset", "csv")
        test_files = [join(test_dir, f) for f in listdir(test_dir)
                       if isfile(join(test_dir, f)) and not f.startswith(".")]
        test_out_dir = join(nfold_output, fold, "testset")
        makedirs(test_out_dir)
        for test_file in test_files:
            converters.nlub_to_hrc2(test_file, test_out_dir, verbose=False)


if __name__ == "__main__":
    arg_parser = init_argparse()
    args = arg_parser.parse_args()
    convert_nfold(args.input, args.out)
