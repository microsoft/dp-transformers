# dp-transformers

:warning: This repo is intended for research projects and prototypes.
While we try to provide tests for all the functionality, the repo has not (yet) undergone the detailed review process that is necessary for deploying a system of critical nature such as privacy.

## Introduction 

See [dp-transformers](https://www.microsoft.com/en-us/research/project/dp-transformers) for a brief introduction to our repository.

## Installation

For installing the `dp-transformers` package, you can just type

```
pip install .
```

## Examples

See `./examples` for end to end examples of how to use the library.

A basic example can be found in `examples/nlg-reddit/sample-level-dp/fine-tune-dp.py`.
First, create an Anaconda environment by doing `conda env create -f examples/nlg-reddit/sample-level-dp/environment.yml`.
Then, you can run the example using the following command (here we assume there are 16 GPUs in the machine, and thus set `--nproc_per_node 16`):

```
python -m torch.distributed.run --nproc_per_node 16 examples/nlg-reddit/sample-level-dp/fine-tune-dp.py \
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
--dataloader_num_workers 2
```

## 🤗 Transformers with Opacus

### Trainer

Huggingface's trainer provides callback hooks which we can use to make sure the required methods in the privacy engine are called.

You can use the callback as demonstrated in the example below

``` python
privacy_engine = opacus.PrivacyEngine(module=model, ...)

# No need to attach the privacy engine to the optimizer. The callback will automatically attach the optimizer.

trainer = transformers.Trainer(
    model = model,
    [...],
    callbacks = [dp_transformers.PrivacyEngineCallback(privacy_engine)]  # <-- Add this line to make sure the privacy engine is used in the trainer
    [...]
)
```

### Data Collation

🤗 Transformers library often provides sensible default arguments.
For example, when no `position_ids` are provided, the library automatically will use incrementing integers.
The way this is implemented is by first creating a tensor of shape `[1, sequence_length]` filled with increasing integers.
During a second step that tensor is replicated for the whole batch.
However, the replication is part of the computational graph and hence Opacus cannot infer the batch size from this input tensor.

We have therefore implemented a custom data collator (see `dp_transformers.DataCollatorForPrivateCausalLanguageModeling`) which automatically creates the `position_ids` input tensor by using `torch.repeat`.
This works with opacus since the `position_ids` tensor appears as batch size different inputs in the computation graph.

### GPT2

The 🤗 Transformers implementation for GPT2 uses a custom layer type namely `Conv1D`.
It is not quite clear why this was introduced since it is essentially a regular linear layer.
This causes problems with Opacus though since it is not sure how to apply the backward hooks for this layer.

In this repo we provide an implementation for handling this type of layer.
See `dp_transformers.grad_sample.transformers.conv_1d`

All necessary grad samplers can be registered by calling `dp_transformers.register_grad_sampler_gpt2()` before the model training.
See the Known Issues section below for more details.

## General tips for DP training

In this section, we collect a few helpful strategies for training models with DP.
Also Opacus's FAQs have a few tips on how to get started with DP training (see [Opacus FAQ](https://opacus.ai/docs/faq))

### Hyper-parameters

Larger batch sizes help DP training.
As a general rule, try starting with $\sqrt{|D|}$ where $D$ is the training dataset.
Since Opacus increases memory consumption significantly, this is only possible using gradient accumulation.

We have found a surprisingly small dependence on the clipping norm.
As a general rule of thumb start with a clipping parameter of 0.1

Fine-tuning the model longer is also helpful.


### Deploying DP trained models

Pay attention which pseudo random number generator (PRNG) was used.
Pytorch's default (Mersenne Twister) might be attackable.
See [Opacus FAQ](https://opacus.ai/docs/faq#what-is-the-secure_rng-argument-in-privacyengine)
Make sure to use a better PRNG before deploying models.

## Known issues

### Register custom grad samplers late in the training process

When registering custom grad sampler like `dp_transformers.register_grad_sampler_gpt2()`, functions are added to a global dictionary that Opacus handles.
This global dictionary is used to establish whether models are compatible with Opacus and how to handle the per-sample gradient computation.
All grad samplers need to be registered as early as possible in the training process.
Definitely, before a `PrivacyEngine` is instantiated.

## How to Cite

```
@misc{dp-transformers,
  author        = {Lukas Wutschitz and Huseyin A. Inan and Andre Manoel},
  title         = {dp-transformers: Training transformer models with differential privacy},
  year          = {2022},
  month         = {August},
  howpublished  = {\url{https://www.microsoft.com/en-us/research/project/dp-transformers}}
}
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

For any other questions, feel free to open an issue on GitHub.
