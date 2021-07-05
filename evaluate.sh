#!/bin/bash
for i in {1..10}
do
    python main.py -n hermit --units 200 --optimizer rmsprop --dropout 0.8 -m testing --run-folder evaluation --train-set "datasets/nlu_benchmark_hrc2/KFold_$i/trainset" --test-set "datasets/nlu_benchmark_hrc2/KFold_$i/testset" -g 0
done
