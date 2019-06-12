
export BERT_BASE_DIR=$1
export DATA_DIR=$2
BASEDIR=$3
SAVE_DIRNAME=$4
MaxLen=$5
TRAIN_FILE=$6
DEV_FILE=$7
TEST_FILE=$8

#source activate cyk
#source activate /export/cyk/envs


python3 ${BASEDIR}/model/bert/run_classification.py \
  --task_name=domainclf \
  --train_file= ${TRAIN_FILE}\
  --dev_file= ${DEV_FILE}\
  --test_file= ${TEST_FILE}\
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=${MaxLen} \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=${BASEDIR}/${SAVE_DIRNAME}_maxLen${MaxLen}/