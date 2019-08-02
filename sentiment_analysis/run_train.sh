
python train.py \
    --do_train \
    --batch_size 64 \
    --model_name 'bert_spc' \
    --dataset 'electric-toothbrush' \
    --device 'cuda:2' \
    --epochs '10' \
    --learning_rate '5e-5' 
    
