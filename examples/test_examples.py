import pytest
import os
import json
import configparser

from subprocess import check_output
from pathlib import Path
from typing import Dict, Union
from azureml.core import Workspace, Run
from dataclasses import dataclass
from datetime import timedelta, datetime


@pytest.fixture(scope="session")
def az_workspace() -> Workspace:
    subscription_id = json.loads(check_output(["az", "account", "show", "--query", "id"]))

    output = json.loads(check_output(["az", "configure", "--list-defaults"]))
    resource_group = next(item for item in output if item["name"] == "group")["value"]
    workspace_name = next(item for item in output if item["name"] == "workspace")["value"]

    workspace = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    return workspace


def submit_example_and_wait_for_metrics(ws: Workspace, aml_config_path: Path) -> Dict[str, Union[float, int]]:
    raw_output = check_output(["az", "ml", "job", "create", "--file", aml_config_path])
    output = json.loads(raw_output)
    run = Run.get(ws, run_id=output["name"])
    try:
        run.wait_for_completion()
    except KeyboardInterrupt as e:
        run.cancel()
        raise e

    metrics = run.get_metrics()

    details = run.get_details()

    metrics["runtime"] = (
        datetime.strptime(details["endTimeUtc"], '%Y-%m-%dT%H:%M:%S.%fZ') -
        datetime.strptime(details["startTimeUtc"], '%Y-%m-%dT%H:%M:%S.%fZ')
    )

    return metrics


@dataclass
class ExampleTest:
    aml_config_path: Path
    expected_trn_loss: float
    expected_val_loss: float
    expected_time: timedelta


@pytest.mark.parametrize("example_test", [
    ExampleTest(
        aml_config_path=Path("examples")/"nlg-reddit"/"author-level-dp"/"aml"/"fuft-eps_8.yml",
        expected_trn_loss=3.72,
        expected_val_loss=3.58,
        expected_time=timedelta(minutes=39),
    ),
    ExampleTest(
        aml_config_path=Path("examples")/"nlg-reddit"/"author-level-dp"/"aml"/"peft-eps_8.yml",
        expected_trn_loss=3.72,
        expected_val_loss=3.58,
        expected_time=timedelta(minutes=39),
    ),
    ExampleTest(
        aml_config_path=Path("examples")/"nlg-reddit"/"sample-level-dp"/"aml"/"fuft-eps_8.yml",
        expected_trn_loss=3.72,
        expected_val_loss=3.58,
        expected_time=timedelta(minutes=39),
    ),
    ExampleTest(
        aml_config_path=Path("examples")/"nlg-reddit"/"sample-level-dp"/"aml"/"fuft-eps_inf.yml",
        expected_trn_loss=3.72,
        expected_val_loss=3.58,
        expected_time=timedelta(minutes=39),
    ),
    ExampleTest(
        aml_config_path=Path("examples")/"nlg-reddit"/"sample-level-dp"/"aml"/"peft-eps_8.yml",
        expected_trn_loss=3.72,
        expected_val_loss=3.58,
        expected_time=timedelta(minutes=39),
    ),
    ExampleTest(
        aml_config_path=Path("examples")/"nlg-reddit"/"sample-level-dp"/"aml"/"peft-eps_inf.yml",
        expected_trn_loss=3.72,
        expected_val_loss=3.58,
        expected_time=timedelta(minutes=39),
    ),
])
def test_nlg_reddit(az_workspace, example_test: ExampleTest):
    metrics = submit_example_and_wait_for_metrics(az_workspace, aml_config_path=example_test.aml_config_path)

    assert metrics["train_loss"] == pytest.approx(example_test.expected_trn_loss, abs=0.02)
    assert metrics["eval_loss"][-1] == pytest.approx(example_test.expected_val_loss, abs=0.02)
    assert abs(metrics["runtime"] - example_test.expected_time) < timedelta(minutes=3)
