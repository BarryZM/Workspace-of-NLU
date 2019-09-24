dataset='frying-pan'
#dataset='vacuum-cleaner'

python train.py \
    --do_train \
    --batch_size 64 \
    --model_name 'bert_spc' \
    --dataset ${dataset} \
    --device 'cuda:2' \
    --epochs '10' \
    --learning_rate '5e-5' 
    
