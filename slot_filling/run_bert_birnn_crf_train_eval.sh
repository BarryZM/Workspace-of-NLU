dataset_name="air-purifier"
type_name='entity'
gpu='3'
epoch=10
max_seq_len=512
hidden_layer=4
target_folder="./outputs/"${dataset_name}_${type_name}_epoch_${epoch}_hidden_layer_${hidden_layer}_max_seq_len_${max_seq_len} 
train_flag=True
eval_flag=False
predict_flag=True


if [ "$type_name" == 'emotion' ] ;then
label_list="O,[CLS],[SEP],B-positibve,I-positibve,B-negative,I-negative,B-moderate,I-moderate"
echo $label_list
fi

if [ "$type_name" == 'entity' ] ;then
label_list="O,[CLS],[SEP],B-3,I-3"
echo $label_list
fi

echo ${target_folder}

if [ $train_flag == True ] ;then
/bin/rm -rf $target_folder
mkdir $target_folder

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
    --train_batch_size=16   \
    --learning_rate=2e-5   \
    --num_train_epochs=$epoch   \
    --output_dir=$target_folder
fi

if [ $predict_flag == True ] ;then
# delete lines which contain [CLS], [SEP]  
cp ${target_folder}/entity_test_results.txt ${target_folder}/entity_test_results.txt-1
sed -i '/SEP/d' ${target_folder}/entity_test_results.txt
sed -i '/CLS/d' ${target_folder}/entity_test_results.txt

python evals/evaluate.py \
    --true_text_path=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/air-purifier/label/test-text.txt \
    --true_label_path=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/comment/air-purifier/label/test-label.txt \
    --predict_label_path=${target_folder}/entity_test_results.txt 
fi
