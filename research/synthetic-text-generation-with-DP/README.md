We present the code of our paper "Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe" at ACL 2023.

## Fine-tuning with DP

The following script assumes distributed training on 8 GPUs.

```console
python -m torch.distributed.run --nproc_per_node 8 fine-tune-dp.py \
    --data_dir $DATA \
    --output_dir $OUTPUT_DIR \
    --model_name gpt2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 64 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --target_epsilon 4.0 \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 50 \
    --logging_steps 10 \
    --max_grad_norm 0 \
    --sequence_len 128 \
    --learning_rate 0.0001 \
    --lr_scheduler_type constant \
    --dataloader_num_workers 2 \
    --disable_tqdm True \
    --load_best_model_at_end True \
```

## Fine-tuning without DP

The following script assumes distributed training on 8 GPUs.

```console
python -m torch.distributed.run --nproc_per_node 8 fine-tune-nodp.py \
    --data_dir $DATA \
    --output_dir $OUTPUT_DIR \
    --model_name gpt2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 64 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 5 \
    --logging_steps 2400 \
    --max_grad_norm 0 \
    --sequence_len 128 \
    --learning_rate 0.00005 \
    --lr_scheduler_type constant \
    --dataloader_num_workers 2 \
    --disable_tqdm True \
    --load_best_model_at_end True \
```

## Synthetic Text Generation

The following script generates synthetic data from a fine-tuned model on a single GPU.

```console
python generate-text.py \
    --model_type gpt2 \
    --model_name_or_path $CHECKPOINT_FOLDER \
    --input_training_file $TRAINING_DATA_FILE \
    --output_dir $OUTPUT_DIR \
    --length 128 \
    --total_sequences 100000 \
    --do_sample \
    --batch_size 8 \
```

## Classification model

The following script assumes distributed training on 8 GPUs. 
Set --sample_dataset True to train the classifier on the original data to sample 100000 data points.

```console
python -m torch.distributed.run --nproc_per_node 8 run-classification.py \
    --model_name_or_path roberta-base \
    --output_dir $OUTPUT_DIR \
    --train_file $TRAINING_DATA_FILE \
    --validation_file $VAL_DATA_FILE \
    --test_file $TEST_DATA_FILE \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 64 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --overwrite_output_dir \
    --overwrite_cache True \
    --evaluation_strategy steps \
    --eval_steps 31 \
    --save_steps 31 \
    --load_best_model_at_end True \
    --label_column_name "label1" \
    --sample_dataset False \
    --disable_tqdm True
```

## Using LoRA during fine-tuning

Although not used in the paper, LoRA fine-tuning significantly improves the runtime by allowing much larger
batch sizes to fit in each GPU. A starting point could be to add `--lora_dim 4 --lora_alpha 32 --lora_dropout 0.0`
and use larger learning rates such as `--learning_rate 3e-4` or `4e-4`.
