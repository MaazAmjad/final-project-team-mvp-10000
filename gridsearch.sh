#!/usr/bin/env bash
python3 gridsearch.py bert-large-uncased gridsearch-ngrams-test-too --max_seq_len 100 101 1 --learning_rate 5e-6 51e-7 2e-7 --num_train_epochs 100 101 1 --permute_ngrams=[1,2,3,4,5,10,20,50]
