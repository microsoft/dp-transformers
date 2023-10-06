import pytest
import os
import json

from subprocess import check_call, check_output
from pathlib import Path
from typing import Dict
from azureml.core import Workspace, Run


@pytest.fixture(scope="session")
def setup_az_workspace() -> Workspace:
    workspace = Workspace(
        subscription_id=os.environ["AZ_SUBSCRIPTION_ID"],
        resource_group=os.environ["AZ_RESOURCE_GROUP"],
        workspace_name=os.environ["AZ_WORKSPACE_NAME"],
    )

    requires_login = False
    if requires_login:
        check_call([
            "az", "login", "--service-principal", "-u", "<app-id>", "-p", "<password-or-cert>", "--tenant", "<tenant>"
        ])
    check_call(["az", "account", "set", "--subscription", workspace.subscription_id])
    check_call(["az", "configure", "--defaults", f"workspace={workspace.name}", f"group={workspace.resource_group}"])

    return workspace


def submit_example_and_wait_for_metrics(ws: Workspace, example_path: Path) -> Dict:
    raw_output = check_output(["az", "ml", "job", "create", "--file", example_path])
    output = json.loads(raw_output)
    run = Run.get(ws, run_id=output["name"])

    run.wait_for_completion()

    metrics = run.get_metrics()

    breakpoint()
    return metrics


class TestExamples:
    @pytest.mark.parametrize("example_path", [
        "peft-eps_inf.yml"
    ])
    def test_nlg_reddit_sample_level_dp(self, setup_az_workspace, example_path: str):
        ws = setup_az_workspace
        metrics = submit_example_and_wait_for_metrics(ws, Path("examples")/"nlg-reddit"/"sample-level-dp"/"aml"/example_path)
        pass
