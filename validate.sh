#!/usr/bin/env bash
python3 run_classifier.py --data_dir ../semeval/ --bert_model bert-large-uncased --task_name semeval --output_dir validation_results --max_seq_length 1 --do_eval --do_lower_case --eval_batch_size 16 --model_path trained_large_model/model.pth
