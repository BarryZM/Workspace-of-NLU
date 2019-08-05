# coding:utf-8
# author:Apollo2Mars@gmail.com

##########
#setting
##########
dataset_name="shaver"

corpus_predict_path=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/${dataset_name}/slot/predict.txt
corpus_format_path=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/${dataset_name}/absa_clf/predict-term-category.txt

ner_result_path=./outputs/${dataset_name}/result_ner.txt
clf_result_path=./outputs/${dataset_name}/result_clf.txt
absa_result_path=./outputs/${dataset_name}/result_absa.txt

if [ $dataset_name = 'shaver' ];then
absa_model='/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/sentiment_analysis/outputs/bert_spc_shaver_val_f10.9156'  
fi

# build folder

if [ ! -d outputs/${dataset_name} ];then
mkdir outputs/${dataset_name}
touch ${ner_result_path}
fi


##########
#NER
##########
# run ner script and post processing
type_name='entity'
gpu='1'

epoch=20
max_seq_len=128
max_seq_len_predict=512
learning_rate=5e-5
hidden_layer=6

target_folder="/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/lexical_analysis/outputs/"${dataset_name}_${type_name}_epoch_${epoch}_hidden_layer_${hidden_layer}_max_seq_len_${max_seq_len}_gpu_${gpu}

if [ "$type_name" == 'entity' ] ;then
label_list="O,[CLS],[SEP],B-3,I-3"
echo $label_list
fi


### Just Predict
#python ../lexical_analysis/models/BERT_BIRNN_CRF.py \
#    --task_name="NER"  \
#    --type_name=${type_name} \
#    --label_list=${label_list} \
#    --gpu=${gpu} \
#    --do_lower_case=False \
#    --do_train=False   \
#    --do_eval=False   \
#    --do_predict=True \
#    --data_dir=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/${dataset_name}/slot   \
#    --vocab_file=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/vocab.txt  \
#    --bert_config_file=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/bert_config.json \
#    --init_checkpoint=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/bert_model.ckpt   \
#    --max_seq_length=$max_seq_len   \
#    --train_batch_size=16   \
#    --learning_rate=${learning_rate}   \
#    --num_train_epochs=$epoch   \
#    --output_dir=$target_folder
#
## delete lines which contain [CLS], [SEP]
#echo ${target_folder}/${type_name}_predict_results.txt
#cp ${target_folder}/${type_name}_predict_results.txt ${ner_result_path}
#sed -i '/SEP/d' ${ner_result_path}
#sed -i '/CLS/d' ${ner_result_path} 
#
### load NER result, make input format file for ABSA and Multi-Level CLF
#
#echo $corpus_predict_path
#echo $ner_result_path

python 1_build_absa_data.py \
    --predict_text_path $corpus_predict_path \
    --entity_result_path ${ner_result_path} \
    --output_path ${corpus_format_path}

##########
## CLF 
##########
CUDA_VISIBLE_DEVICES=2 python ../classification/train.py  \
    --dataset_name ${dataset_name} \
    --model_name text_cnn \
    --epoch 5 \
    --batch_size 64 \
    --do_predict \
    --learning_rate 1e-3 \
    --results_file ${clf_result_path} \
    --outputs_folder /export/home/sunhongchao1/1-NLU/Workspace-of-NLU/classification/outputs/${dataset_name} 

##########
## absa
##########
CUDA_VISIBLE_DEVICES=2 python ../sentiment_analysis/train.py \
    --dataset ${dataset_name} \
    --model_name 'bert_spc' \
    --batch_size 64 \
    --do_predict \
    --device 'cuda:0' \
    --epochs '30' \
    --results_file ${absa_result_path} \
    --load_model_path ${absa_model}

python 2_all_results.py \
    --input_path ${corpus_format_path}\
    --result_clf ${clf_result_path}\
    --result_absa ${absa_result_path}\
    --output_path result_all.txt
