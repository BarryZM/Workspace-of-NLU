dataset_name='frying-pan'
#dataset_name='vacuum-cleaner'
#dataset_name='electric-toothbrush'
model_name='text_cnn'
epoch='30'
batch_size='64'
#output_folder=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/classification/outputs/${dataset_name}_epoch_${epoch}_batch_name_${batch_size}
output_folder=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/classification/outputs/${dataset_name}

CUDA_VISIBLE_DEVICES=1 python train.py --model_name ${model_name} \
                --dataset_name ${dataset_name} \
                --do_train \
                --do_test \
                --epoch ${epoch} \
                --batch_size ${batch_size} \
                --outputs_folder ${output_folder}
 
