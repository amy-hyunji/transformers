CUDA_VISIBLE_DEVICES=1,3,4,6 python run_language_modeling.py --tokenizer_name=bert-base-uncased --fp16 --output_dir=output --do_train --train_data_file=/home/hyunji/Electra-pretraining/dataset/owt_txt --model_type=electra --per_device_train_batch_size=1 --num_train_epochs=40 --save_steps=10000 --max_steps=1000000 --block_size=128
