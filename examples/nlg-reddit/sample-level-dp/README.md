# Differentially private fine-tuning of a GPT-2 style model

This example fine-tunes generative language models (such as GPT-2 series) with Differential Privacy on a text corpus.
In this case 500,000 samples of Reddit comments.
We compare different fine-tuning techniques (full fine-tuning, LoRA) and also provide a data distributed implementation for faster training.
These merely serve as examples as hyperparameters are not optimized and corresponding commands are presented below.

# Results

| Model (HF) | Fine-tuning Method | DP  | GPUs   | Epochs | Train Loss | Eval Loss | $\varepsilon$ | Run Time [s] | AML Config            |
| ---------- | ------------------ | --- | ------ | ------ | ---------- | --------- | ------------- | ------------ | --------------------- |
| gpt2       | Full               | Yes | 8xV100 |    3   |    3.75    |   3.61    |      8.0      |    1944      | fuft-eps_8.yml        |
| gpt2       | Full               | No  | 8xV100 |    3   |    3.56    |   3.46    | -             |    1227      | fuft-no_inf.yml       |
| gpt2       | LoRA               | Yes | 8xV100 |    3   |    3.74    |   3.60    |      8.0      |    1128      | peft-eps_8.yml        |
| gpt2       | LoRA               | Yes | 1xV100 |    3   |    3.74    |   3.60    |      8.0      |    12248     | peft-eps_8-gpus_1.yml |
| gpt2       | LoRA               | No  | 8xV100 |    3   |    3.70    |   3.58    | -             |    1006      | peft-eps_8.yml        |


## Azure Machine Learning

We provide Azure Machine Learning (AML) configuration files for the above experiments.

```
az ml job create --file aml/<aml config>
```


## Local Training

Alternatively, you can run the training script directly on your local machine.

Install the environment (assuming CUDA 11.6) with

```
conda env create -f environment.yml
conda activate dp-transformers
```

And run one of the following training scripts.

### Fine-tune the full model with DP

```
python -m torch.distributed.run --nproc_per_node 16 fine-tune-dp.py \
--output_dir scratch \
--model_name gpt2 \
--sequence_len 128 \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 2 \
--evaluation_strategy steps \
--eval_steps 45 \
--log_level info \
--per_device_eval_batch_size 64 \
--eval_accumulation_steps 1 \
--seed 42 \
--target_epsilon 8 \
--per_sample_max_grad_norm 1.0 \
--prediction_loss_only \
--weight_decay 0.01 \
--remove_unused_columns False \
--num_train_epochs 3 \
--logging_steps 5 \
--max_grad_norm 0 \
--lr_scheduler_type constant \
--learning_rate 1e-4 \
--disable_tqdm True \
--dataloader_num_workers 2 \
--label_names labels
```

### Fine-tune the full model without DP

```
python -m torch.distributed.run --nproc_per_node 16 fine-tune-nodp.py \
--output_dir scratch \
--model_name gpt2 \
--sequence_len 128 \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--evaluation_strategy steps \
--eval_steps 45 \
--log_level info \
--per_device_eval_batch_size 64 \
--eval_accumulation_steps 1 \
--seed 42 \
--prediction_loss_only \
--weight_decay 0.01 \
--remove_unused_columns False \
--num_train_epochs 3 \
--logging_steps 5 \
--max_grad_norm 0 \
--lr_scheduler_type constant \
--learning_rate 2e-4 \
--disable_tqdm True \
--dataloader_num_workers 2 \
--label_names labels
```

### Fine-tune only the LoRA layers introduced into the model with DP

```
python -m torch.distributed.run --nproc_per_node 16 fine-tune-dp.py \
--output_dir scratch \
--model_name gpt2 \
--sequence_len 128 \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--evaluation_strategy steps \
--eval_steps 45 \
--log_level info \
--per_device_eval_batch_size 64 \
--eval_accumulation_steps 1 \
--seed 42 \
--target_epsilon 8 \
--per_sample_max_grad_norm 1.0 \
--prediction_loss_only \
--weight_decay 0.01 \
--remove_unused_columns False \
--num_train_epochs 3 \
--logging_steps 5 \
--lora_dim 4 \
--lora_alpha 32 \
--lora_dropout 0.0 \
--max_grad_norm 0 \
--lr_scheduler_type constant \
--learning_rate 3e-4 \
--disable_tqdm True \
--dataloader_num_workers 2 \
--label_names labels
```

### Fine-tune only the LoRA layers introduced into the model without DP

```
python -m torch.distributed.run --nproc_per_node 16 fine-tune-nodp.py \
--output_dir scratch \
--model_name gpt2 \
--sequence_len 128 \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--evaluation_strategy steps \
--eval_steps 45 \
--log_level info \
--per_device_eval_batch_size 64 \
--eval_accumulation_steps 1 \
--seed 42 \
--prediction_loss_only \
--weight_decay 0.01 \
--remove_unused_columns False \
--num_train_epochs 3 \
--logging_steps 5 \
--lora_dim 4 \
--lora_alpha 32 \
--lora_dropout 0.0 \
--max_grad_norm 0 \
--lr_scheduler_type constant \
--learning_rate 5e-4 \
--disable_tqdm True \
--dataloader_num_workers 2 \
--label_names labels
```