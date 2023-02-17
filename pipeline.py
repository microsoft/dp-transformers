from azure.ml.component import Component
from azure.ml.component import dsl
from azureml.core import Datastore, Dataset, Workspace


TRAINING_DATA = {
    "name": "Cloverport_PublicGroupEmail_6months_b75af18b-3415-aa3d-947e-3b67741a13e5_File",
    "version": "1"
}


aml_config = {
    "subscription_id": "95e0ae98-c287-4f51-b23e-4e2317b7e169",
    "resource_group": "EYESON.HERON.PROD.7d040e3f-0aae-465a-96c4-c7583aa3faf5",
    "workspace_name": "amlworkspacen5udte35numwq"
}

ws = Workspace.get(name=aml_config['workspace_name'], subscription_id=aml_config['subscription_id'], resource_group=aml_config['resource_group'])

# this is standard way to get workspace but I didn't want to share too many files, I added aml_config.json in the same file
#ws = Workspace.from_config() # if you have config.json in the same folder

#preprocessing_module = Component.load(name="data_preprocess", version="0.0.19", workspace=ws)
preprocessing_module = Component.load(name="data_preprocess_HasAttachments", version="0.0.0", workspace=ws)
#preprocessing_module = Component.from_yaml(yaml_file="preprocess_spec.yaml")
split_module = Component.load(name="data_split", version="0.0.0", workspace=ws)
#split_module = Component.from_yaml(yaml_file="split_spec.yaml")
dp_module = Component.from_yaml(yaml_file="train_spec.yaml")
generation_module = Component.from_yaml(yaml_file="generate_spec.yaml")
classification_module = Component.from_yaml(yaml_file="classification_spec.yaml")

training_dataset = Dataset.get_by_name(ws, name=TRAINING_DATA['name'], version=TRAINING_DATA['version'])


# Create a pipeline with a single component
@dsl.pipeline(
    name="DP-Transformer Training Pipeline",
    description="DP-Transformer Training Pipeline",
    default_compute_target="lin-gpu-6")
def dp_transformer_training_pipeline(
    training_dataset: Dataset):
    """
    Creates a pipeline 
    """

    preprocessing_module_step = preprocessing_module(
        training_data=training_dataset
    )
    #preprocessing_module_step.runsettings.configure(target="lin-gpu-6")
    #preprocessed_data = preprocessing_module_step.outputs.output_data

    split_module_step = split_module(
        training_data=preprocessing_module_step.outputs.output_dir
    )

    classification_module_step_1 = classification_module(
        train_file=split_module_step.outputs.output_dir,
        validation_file=split_module_step.outputs.output_dir,
        test_file=split_module_step.outputs.output_dir
    )

    dp_module_step = dp_module(
        training_data=split_module_step.outputs.output_dir
    )

    generation_module_step = generation_module(
        input_training_file=split_module_step.outputs.output_dir,
        model_name_or_path=dp_module_step.outputs.output_dir
    )

    classification_module_step_2 = classification_module(
        train_file=generation_module_step.outputs.output_dir,
        validation_file=split_module_step.outputs.output_dir,
        test_file=split_module_step.outputs.output_dir
    )

    return 

if __name__ == "__main__":

    pipeline_instance = dp_transformer_training_pipeline(training_dataset)
    pipeline_instance.validate(workspace=ws)
    pipeline_instance.submit(workspace=ws, experiment_name="debug-train")