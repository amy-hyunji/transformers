CUDA_VISIBLE_DEVICES=1,2,3 python3 run_language_modeling.py --tokenizer_name=bert-base-uncased --fp16 --output_dir=output --do_train --train_data_file=./owt_tensor_nsml --model_type=electra --per_device_train_batch_size=1 --num_train_epochs=40 --save_steps=10000 --max_steps=1000000 --block_size=128
