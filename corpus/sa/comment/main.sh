# Convert origin file to train format data
#dataset_name='frying-pan'
#dataset_name='vacuum-cleaner'
dataset_name='shaver'
#dataset_name='electric-toothbrush'
type_name='entity'

do_train_test_convert=True
do_predict_convert=True

train_dir=./${dataset_name}/train_csv
test_dir=./${dataset_name}/test_csv
predict_dir=./${dataset_name}/predict_text

echo $train_dir
echo $test_dir
echo $predict_dir

if [ ${do_train_test_convert} == True ];then
python 1_lexical_train-test.py \
    --dataset_name ${dataset_name} \
    --train_dir ${train_dir} \
    --test_dir ${test_dir} \
    --output_dir ${dataset_name}/slot

python 2_absa-clf_train-test.py \
    --dataset_name ${dataset_name} \
    --train_dir ${train_dir} \
    --test_dir ${test_dir} \
    --output_dir ${dataset_name}/absa_clf
fi

if [ ${do_predict_convert} == True ];then
python 3_lexical_predict.py \
    --dataset_name ${dataset_name} \
    --predict_dir ${predict_dir} \
    --output_dir ${dataset_name}/slot

#python 4_absa-clf_predict.py \
#   --dataset_name ${dataset_name} \
#   --predict_dir ${predict_dir} \
#   --output_dir ${dataset_name}/absa_clf
fi

