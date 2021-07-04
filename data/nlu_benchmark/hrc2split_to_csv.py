import argparse
import pandas as pd
from os import listdir
from os.path import isfile, join, isdir
from progress.bar import Bar


"""
    script to create a two csv files (train.csv, test.csv) in the NLU-benchmark
    format, starting from a fixed split of hrc2 files and the whole NLU-benchmark
    file.
"""

parser = argparse.ArgumentParser(description='hrc2 fixed split to NLU Benchmark train and test in csv')
parser.add_argument('input_split_dir', type=str, default=None,
                    help='Directory containing the fixed split')
parser.add_argument('nlu_bench_csv', type=str, default=None,
                    help='NLU-Benchmark csv file')
parser.add_argument('-o', '--output', type=str, default=".",
                    help='Output dir')
args = parser.parse_args()

split_dir = args.input_split_dir
nlu_benchmark_file = args.nlu_bench_csv
output_dir = args.output

nlu_benchmark = pd.read_csv(nlu_benchmark_file, sep=";")
train_set_df = pd.DataFrame(columns=list(nlu_benchmark))
test_set_df = pd.DataFrame(columns=list(nlu_benchmark))

train_set_dir = join(split_dir, "train")
test_set_dir = join(split_dir, "test")

train_files = [f for f in listdir(train_set_dir) if isfile(join(train_set_dir, f)) and not f.startswith(".")]
test_files = [f for f in listdir(test_set_dir) if isfile(join(test_set_dir, f)) and not f.startswith(".")]

#bar = Bar('Loading train', max=len(train_files))
#for f in train_files:
#    nlu_bench_id = int(f.split("_")[0])
#    train_set_df = train_set_df.append(nlu_benchmark.loc[nlu_bench_id])
#    bar.next()
#bar.finish()

bar = Bar('Loading test', max=len(test_files))
for f in test_files:
    nlu_bench_id = int(f.split("_")[0])
    test_set_df = test_set_df.append(nlu_benchmark.loc[nlu_bench_id])
    bar.next()
bar.finish()

print("Saving csv files")
train_set_df.to_csv(join(output_dir, "train.csv"), sep=";", index=False)
test_set_df.to_csv(join(output_dir, "test.csv"), sep=";", index=False)
