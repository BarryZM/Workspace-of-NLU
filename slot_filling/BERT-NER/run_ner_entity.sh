mkdir ./output/entity_epoch_20_hidden_layer_4

python BERT_NER_entity.py\
    --task_name="NER"  \
    --gpu='1' \
    --do_lower_case=False \
    --do_train=False   \
    --do_eval=False   \
    --do_predict=True \
    --data_dir=data   \
    --vocab_file=chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=64   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=20.0   \
    --output_dir=./output/entity_epoch_20_hidden_layer_4
