#!/bin/bash
python3 bert_kash.py --train_set ../1-data/train-new.txt --batch_size 50 --validate_set ../1-data/test-31k.txt --test_set ../1-data/test-31k.txt --epoch 50 --model_save_path ./bert-emb-CNN-BLSTM-50/ --model_save_name bert-emb-CNN-BLSTM-50.pb --gpu_id 2
