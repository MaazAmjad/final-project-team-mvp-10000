#!/usr/bin/env bash
python3 gridsearch.py bert-large-uncased gridsearch-small-seqlen-vs-learningrate --max_seq_len 50 201 50 --learning_rate 5e-7 3e-6 5e-7 --num_train_epochs 100 101 1
