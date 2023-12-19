# Differentially private fine-tuning of LLMs using QLoRA

We demonstrate examples of fine-tuning Mistral 7B using QLoRA with and without DP.

# Results

| Dataset (HF) | DP  | GPUs   | Epochs | Max Eval Accuracy | $\varepsilon$ | Run Time [s] | AML Config            |
| ---------- | --- | ------ | ------ | --------- | ------------- | ------------ | --------------------- |
| sst2       | Yes | 8xA100 |    3   |   96.44    |      8.0      |    .      | sst2/peft-eps_8.yml        |
| sst2       | No  | 8xA100 |    3   |   97.25    | -             |    .      | sst2/peft-eps_inf.yml       |
| qnli       | Yes | 8xA100 |    3   |   94.80    |      8.0      |    .      | qnli/peft-eps_8.yml        |
| qnli       | No  | 8xA100 |    3   |   96.40    | -             |    .      | qnli/peft-eps_inf.yml       |

| Dataset (HF) | DP  | GPUs   | Epochs | Min Eval Loss | Test ROUGE1 | Test ROUGE2 | Test ROUGEL | $\varepsilon$ | Run Time [s] | AML Config |
| ------------ | --- | ------ | ------ | ------------- | ----------- | ----------- | ----------- | ------------- | ------------ | ---------- |
| cnn          | Yes | 8xA100 |    3   |     0.9624    |   44.14     |    22.09    |   30.90     |      8.0      |    .         | cnn/peft-eps_8.yml |
| cnn          | No  | 8xA100 |    3   |     0.9188    |   45.01     |    22.98    |   31.67     | -             |    .         | cnn/peft-eps_inf.yml |

## Azure Machine Learning

We provide Azure Machine Learning (AML) configuration files for the above experiments.

```
az ml job create --file aml/<aml config>
```


## Local Training

Alternatively, you can run the training script directly on your local machine.

Install the environment with

```
conda env create -f environment.yml
conda activate dp-transformers
```

Follow the training scripts under aml folder.
