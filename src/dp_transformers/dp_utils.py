# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import opacus
from datasets import Dataset
from contextlib import contextmanager
from typing import Sequence

from dp_transformers.trainer import DPTrainer


class GradSampleModule(opacus.GradSampleModule):
    """
    Little wrapper to provide `no_sync` context which is assumed by Huggingface trainer.
    We don't need to do anything in addition here
    """
    @contextmanager
    def no_sync(self):
        yield


def create_author_mapping(dataset: Dataset, author: str) -> Sequence[Sequence[int]]:
    """
    Creates a mapping from authors to samples in a dataset.
    """
    with dataset.formatted_as(type="pandas"):
        authors = pd.DataFrame(data={"author": dataset[author]})
        author_mapping = [g.index.values for _, g in authors.groupby("author")]
    return author_mapping

# For backwards compatibility
OpacusDPTrainer = DPTrainer

