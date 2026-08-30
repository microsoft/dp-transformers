import numpy as np
import torch
from typing import Sequence, Dict, List



class AuthorIndexedDataset:
    def __init__(self, dataset: Sequence, author_index: Dict[int, List[int]], rng: torch.Generator):
        self.dataset = dataset
        self.author_index = author_index
        self.rng = rng

    def __getitem__(self, index):
        # Randomly select a sample from the author's index
        sample_from_author = self.author_index[index][torch.randint(len(self.author_index[index]), (1,), generator=self.rng).item()]
        return self.dataset[sample_from_author]
    
    def __len__(self):
        return len(self.author_index)
