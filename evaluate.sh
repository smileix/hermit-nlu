#!/bin/bash

python main.py -n hermit --units 200 --optimizer rmsprop --dropout 0.8 -m testing --run-folder evaluation --train-set "datasets/nlu_benchmark_hrc2/KFold_1/trainset" --test-set "datasets/nlu_benchmark_hrc2/KFold_1/testset" -g 0,1,2,3

