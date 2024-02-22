import numpy as np
import torch
from typing import Sequence, Dict



class AuthorIndexedDataset:
    def __init__(self, dataset: Sequence, author_index: Dict, rng: torch.Generator):
        self.dataset = dataset
        self.author_index = author_index
        self.rng = rng

    def __getitem__(self, index):
        return self.dataset[torch.randint(len(self.author_index[index]), (1,), generator=self.rng).item()]
    
    def __len__(self):
        return len(self.author_index)
