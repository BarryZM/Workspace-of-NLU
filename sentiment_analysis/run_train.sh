
python train.py \
    --do_train \
    --batch_size 64 \
    --model_name 'bert_spc' \
    --dataset 'air-purifier' \
    --device 'cuda:2' \
    --epochs '50' \
    --valset_ratio 0.1
    
