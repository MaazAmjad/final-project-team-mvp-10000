#!/usr/bin/env bash
python3 run_classifier.py --data_dir semeval/ --bert_model bert-base-uncased --task_name semeval --output_dir models2/ --max_seq_length 240 --do_eval --do_lower_case --eval_batch_size 16 --model_path trained_model/model.pth
