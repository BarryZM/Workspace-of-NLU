# coding:utf-8
# author:Apollo2Mars@gmail.com

###########
# input format
# 
##########
input_mode='file'
# process input file and copy to corpus folder
python convert_text.py --input_file '0_origin.txt'
cp result_convert.txt /export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/air-purifier/label_100test/test.txt
cp result_convert.txt /export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/air-purifier/label_100test/train.txt
cp result_convert.txt /export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/air-purifier/label_100test/dev.txt

##########
#NER
##########
# run ner script and post processing
dataset_name="air-purifier"
type_name='entity'
gpu='0'

epoch=10
max_seq_len=128
max_seq_len_predict=128
learning_rate=5e-5
hidden_layer=4
target_folder="/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/slot_filling/outputs/"${dataset_name}_${type_name}_epoch_${epoch}_hidden_layer_${hidden_layer}_max_seq_len_${max_seq_len}_gpu_${gpu}
train_flag=False
eval_flag=False
predict_flag=True
metric_flag=True

if [ "$type_name" == 'entity' ] ;then
label_list="O,[CLS],[SEP],B-3,I-3"
echo $label_list
fi

echo ${target_folder}

if [ $train_flag == True -o $predict_flag == True ] ;then

# Just Predict
python ../slot_filling/models/BERT_BIRNN_CRF.py \
    --task_name="NER"  \
    --type_name=${type_name} \
    --label_list=${label_list} \
    --gpu=${gpu} \
    --do_lower_case=False \
    --do_train=${train_flag}   \
    --do_eval=${eval_flag}   \
    --do_predict=${predict_flag} \
    --data_dir=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/${dataset_name}/label_100test   \
    --vocab_file=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/resources/chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=$max_seq_len   \
    --train_batch_size=16   \
    --learning_rate=${learning_rate}   \
    --num_train_epochs=$epoch   \
    --output_dir=$target_folder
fi

# delete lines which contain [CLS], [SEP]
if [ $metric_flag == True ] ;then
cp ${target_folder}/${type_name}_test_results.txt outputs/${type_name}_test_results.txt
sed -i '/SEP/d' outputs/${type_name}_test_results.txt
sed -i '/CLS/d' outputs/${type_name}_test_results.txt
fi

# load NER result, make input format file for ABSA and Multi-Level CLF

python build_absa_data.py

#########
# CLF 
#########
CUDA_VISIBLE_DEVICES=2 python ../classification/train.py  \
    --dataset_name air-purifier \
    --model_name text_cnn \
    --epoch 5 \
    --batch_size 64 \
    --do_test \
    --learning_rate 1e-3 \
    --label_list "'指示灯', '味道', '运转音', '净化效果', '风量', '电源', '尺寸', '感应', '设计', '滤芯滤网', '模式', '操作', '包装', '显示', '功能', '价保', '发票', '商品复购', '商品用途', '商品价格', '商品质量', '商品颜色', '商品外观', '商品营销', '商品品牌', '商品产地', '商品其他', '客服态度', '客服处理速度', '客服其他', '配送速度', '物流态度', '物流其他', '维修服务', '安装服务', '退货服务', '换货服务', '质保', '退款服务', '售后其他'" \
    --results_file result_clf_text.txt \
    --outputs_folder /export/home/sunhongchao1/1-NLU/Workspace-of-NLU/classification/outputs/ 
#
# post processing


#########
# ABSA
#########
python ../sentiment_analysis/train.py \
    --batch_size 64 \
    --model_name 'bert_spc' \
    --dataset 'air-purifier-100-test' \
    --device 'cuda:0' \
    --epochs '30' \
    --do_test \
    --results_file result_absa_text.txt
#    --outputs_folder /export/home/sunhongchao1/1-NLU/Workspace-of-NLU/classification/outputs/ \ 
 
# post processing

