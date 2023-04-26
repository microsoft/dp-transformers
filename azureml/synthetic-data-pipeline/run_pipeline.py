import yaml
from azure.ml.component import Component
from azure.ml.component import dsl
from azureml.core import Dataset, Workspace


# Load settings
with open("aml_config.yaml", "r") as settings_file:
    config = yaml.safe_load(settings_file)

AML_CONFIG = config['aml_config']
MODULE_NAMES = config['module_names']
TRAINING_DATA = config['training_data']
DEFAULT_COMPUTE_TARGET = config['default_compute_target']

# Set up workspace and modules
ws = Workspace.get(name=AML_CONFIG['workspace_name'], subscription_id=AML_CONFIG['subscription_id'], resource_group=AML_CONFIG['resource_group'])

preprocessing_module = Component.load(name=MODULE_NAMES['preprocessing'], version="0.0.0", workspace=ws)
split_module = Component.load(name=MODULE_NAMES['split'], version="0.0.0", workspace=ws)
finetuning_module = Component.from_yaml(yaml_file="finetuning_spec.yaml")
generation_module = Component.from_yaml(yaml_file="generation_spec.yaml")
classification_module = Component.from_yaml(yaml_file="classification_spec.yaml")

training_dataset = Dataset.get_by_name(ws, name=TRAINING_DATA['name'], version=TRAINING_DATA['version'])


@dsl.pipeline(
    name="DP-Transformer Training Pipeline",
    description="DP-Transformer Training Pipeline",
    default_compute_target=DEFAULT_COMPUTE_TARGET)
def dp_transformer_training_pipeline(
    training_dataset: Dataset):
    """Creates a pipeline"""

    preprocessing_module_step = preprocessing_module(
        training_data=training_dataset
    )

    split_module_step = split_module(
        training_data=preprocessing_module_step.outputs.output_dir
    )

    classification_nodp = classification_module(
        train_file=split_module_step.outputs.output_dir,
        validation_file=split_module_step.outputs.output_dir,
        test_file=split_module_step.outputs.output_dir
    )

    dp_module_step = finetuning_module(
        training_data=split_module_step.outputs.output_dir
    )

    generation_module_step = generation_module(
        input_training_file=split_module_step.outputs.output_dir,
        model_name_or_path=dp_module_step.outputs.output_dir
    )

    classification_dp = classification_module(
        train_file=generation_module_step.outputs.output_dir,
        validation_file=split_module_step.outputs.output_dir,
        test_file=split_module_step.outputs.output_dir
    )

    return 


if __name__ == "__main__":
    pipeline_instance = dp_transformer_training_pipeline(training_dataset)
    pipeline_instance.validate(workspace=ws)
    pipeline_instance.submit(workspace=ws, experiment_name="debug-train")