# Example of Synthetic Data Generation Pipeline

In this folder, we specify three AzureML components that can help you in creating your own synthetic data generation pipeline:

- `finetuning`: takes a CSV file as input and fine-tunes a LLM with differential privacy
- `generation`: uses the LLM to generate a synthetic dataset
- `classification`: trains and evaluates a classification model on either (part of) the real dataset, or the synthetic dataset, so that we can compare the performance in both

## Template for `aml_config.yaml` file

```yaml
aml_config:
  subscription_id:
  resource_group:
  workspace_name:

module_names:
  preprocessing:
  split:

training_data:
  name:
  version:

default_compute_target:
```