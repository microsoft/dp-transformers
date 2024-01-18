import torch
from transformers import DataCollator, DataCollatorForLanguageModeling, PreTrainedTokenizer
from typing import List, Union, Dict, Mapping, Sequence, Optional


class DataCollatorForPrivateCausalLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def __call__(self, features: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]],
                 return_tensors: Optional[str] = None) -> Dict[str, torch.Tensor]:
        batch = super().__call__(features, return_tensors=return_tensors)

        # Huggingface's default way of constructing position_ids is not compatible with Opacus
        # since Opacus is not able to deduce the batch size from the input. Here we manually
        # generate a position_ids tensor which has the same values as Huggingface's default tensor
        # but it is constructed in a way that is compatile with Opacus by using expand_as.
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


class DataCollatorWithEmptyWrapper:
    def __init__(self, original_collator: DataCollator, sample_empty_shapes: Mapping[str, Sequence[int]], dtypes: Mapping[str, torch.dtype]):
        self.original_collator = original_collator
        self.sample_empty_shapes = sample_empty_shapes
        self.dtypes = dtypes

    @classmethod
    def from_batch(cls, original_collator: DataCollator, batch: Dict[str, torch.Tensor]):
        sample_empty_shapes = {k: (0, *(v.shape[1:])) for k, v in batch.items()}
        dtypes = {k: v.dtype for k, v in batch.items()}
        return cls(original_collator=original_collator, sample_empty_shapes=sample_empty_shapes, dtypes=dtypes)

    def __call__(self, features: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]], return_tensors: Optional[str] = None) -> Dict[str, torch.Tensor]:
        if len(features) > 0:
            return self.original_collator(features, return_tensors=return_tensors)
        else:
            return {
                k: torch.zeros(self.sample_empty_shapes[k], dtype=self.dtypes[k])
                for k in self.sample_empty_shapes
            }

