# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding

from opacus.grad_sample import utils
from dp_transformers.grad_sample.transformers.tied_embedding import compute_tied_embedding_grad_sample


def compute_positional_embedding_grad_sample(
    layer: BartLearnedPositionalEmbedding, input_ids: torch.Size, B: torch.Tensor, batch_dim: int = 0) -> None:
    """
    Computes per sample gradients for BART's ``BartLearnedPositionalEmbedding`` layer.

    This is essentially the same code as ``compute_tied_embedding_grad_sample``, however, we first 
    offset in the layer.num_embedding dimensions to account for padding IDs in BART.

    Args:
        layer: Layer
        A: Activations - torch.Size([batch_size, embed_dim])
        B: Backpropagations - torch.Size([embed_dim, embed_dim])
        batch_dim: Batch dimension position
    """
    batch_size, seq_len = input_ids.shape[:2]
    past_key_values_length = 0 # used for encoder, but may be non-zero for decoder
    positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=layer.weight.device
        ).repeat(batch_size, 1)
        
    A = positions + layer.offset
    
    return compute_tied_embedding_grad_sample(layer, A, B, batch_dim)


def register_grad_sampler() -> None:
    utils.register_grad_sampler(BartLearnedPositionalEmbedding)(compute_positional_embedding_grad_sample)
