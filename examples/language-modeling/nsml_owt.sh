nsml run -d owt -c 10 -g 2 --memory '60G' -m "MP transformer original owt" -e run_language_modeling.py -a "--tokenizer_name=bert-base-uncased --fp16 --output_dir=output --do_train --train_data_file=/data/owt/train --model_type=electra --per_device_train_batch_size=64 --num_train_epoch=40 --save_steps=10000 --max_steps=1000000 --block_size=128" 
