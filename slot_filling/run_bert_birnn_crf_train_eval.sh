dataset_name="air-purifier"
type_name='entity'
gpu='3'
epoch=20.0
max_seq_len=128
hidden_layer=4
target_folder="./outputs/"${dataset_name}_${type_name}_epoch_${epoch}_hidden_layer_${hidden_layer}_max_seq_len_${max_seq_len} 
train_flag=True
eval_flag=False
predict_flag=True


if [ "$type_name" == 'emotion' ] ;then
echo 'bbb'
label_list="O,[CLS],[SEP],B-positibve,I-positibve,B-negative,I-negative,B-moderate,I-moderate"
fi

if [ "$type_name" == 'entity' ] ;then
echo 'aaa'
label_list="O,[CLS],[SEP],B-3,I-3"
fi

echo ${target_folder}

if [ $train_flag == True ] ;then
/bin/rm -rf $target_folder
mkdir $target_folder
fi

python models/BERT_BIRNN_CRF.py \
    --task_name="NER"  \
    --type_name=${type_name} \
    --label_list=${label_list} \
    --gpu=${gpu} \
    --do_lower_case=False \
    --do_train=${train_flag}   \
    --do_eval=${eval_flag}   \
    --do_predict=${predict_flag} \
    --data_dir=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/${dataset_name}/label   \
    --vocab_file=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=$max_seq_len   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=$epoch   \
    --output_dir=$target_folder

# predict 之后的问题处理
# 删除 [CLS]
# 将[SEP] 变空行 


#python evals/evaluate.py \
#    --label2id_path=./output/label2id_entity.pkl \
#    --true_text_path=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/air-purifier/label/test_text.txt\
#    --true_label_path=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/air-purifier/label/test_label.txt\
#    --true_predict_path=${target_folder}/   
