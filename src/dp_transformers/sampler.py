# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from typing import Sequence
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

from typing import Iterator, List

class AuthorSampler(Sampler):
    def __init__(self, author_sampler: Sampler, author_mapping: Sequence[Sequence[int]]):
        self.author_mapping = list(author_mapping)
        self.author_sampler = author_sampler
        self.indices = [0 for _ in range(len(self.author_mapping))]

    def __len__(self) -> int:
        return len(self.author_sampler)

    def __iter__(self) -> Iterator[List[int]]:
        for batch_author_ids in self.author_sampler:
            sample_ids = [self.indices[author_id] for author_id in batch_author_ids]
            for author_id in batch_author_ids:
                self.indices[author_id] += 1
                self.indices[author_id] = self.indices[author_id] % len(self.author_mapping[author_id])
            yield [int(self.author_mapping[author_id][sample_id]) for author_id, sample_id in zip(batch_author_ids, sample_ids)]


class PoissonAuthorSampler(AuthorSampler):
    def __init__(self, author_mapping: Sequence[Sequence[int]], sample_rate: float) -> None:
        """
        Create batches by first sampling authors with uniform probability and then sampling a random element from the author

        :param author_mapping: A mapping where `dataset[author_mapping[i][j]]` produces the j-th sample of the i-th author in the dataset.
        :type author_mapping: Sequence[Sequence[int]]
        :param float sample_rate: Probability with which a author is sampled `E[len(batch_size)] = sample_rate*len(dataset)`
        """
        author_sampler = UniformWithReplacementSampler(
            num_samples=len(author_mapping),
            sample_rate=sample_rate
        )
        super().__init__(author_sampler, author_mapping)


class ShuffledAuthorSampler(AuthorSampler):
    def __init__(self, author_mapping: Sequence[Sequence[int]], batch_size: int, world_size: int) -> None:
        """
        Create batches by first shuffling the authors and then sampling the next element from the author

        :param author_mapping: A mapping where `dataset[author_mapping[i][j]]` produces the j-th sample of the i-th author in the dataset.
        :type author_mapping: Sequence[Sequence[int]]
        :param int batch_size: Batch size of the output
        """
        if world_size <= 1:
            author_sampler = BatchSampler(RandomSampler(author_mapping), batch_size=batch_size, drop_last=True)
        else:
            author_sampler = BatchSampler(DistributedSampler(author_mapping), batch_size=batch_size, drop_last=True)
        super().__init__(author_sampler, author_mapping)