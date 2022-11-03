import torch
from opacus.grad_sample import utils

from dp_transformers.layers.relative_position_embedding import RelativePositionEmbedding


def compute_relative_position_embedding_grad_sample(
        layer: RelativePositionEmbedding, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``RelativePositionEmbedding`` layer.
    This is essentially a wrapper module on nn.Embedding with two key differences:
    1.  It is a tied embedding layer
    2. It is being used specifically in an attention head to find embedding for relative position of query and key

    This is essentially the same code as tied_embedding provides, however, we have to duplicate the gradients for each
    of the example in the batch since the relative position embedding for a batch only depends on the sequence length
    of the batch

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    saved = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True

    A = A.unsqueeze(0).expand(layer.training_batch_size, *A.shape)
    B = B.unsqueeze(0).expand(layer.training_batch_size, *B.shape)

    batch_size = A.shape[batch_dim]
    index = (
        A.unsqueeze(-1)
            .expand(*A.shape, layer.embedding_dim)
            .reshape(batch_size, -1, layer.embedding_dim)
    )
    grad_sample = torch.zeros(
        batch_size, *layer.weight.shape, device=layer.weight.device
    )
    grad_sample.scatter_add_(1, index, B.reshape(batch_size, -1, layer.embedding_dim))
    torch.backends.cudnn.deterministic = saved

    # Overriding the batch length observed based on the training examples
    # with the training batch size received during initialization of the layer
    layer.max_batch_len = layer.training_batch_size
    utils.create_or_accumulate_grad_sample(layer.weight, grad_sample, layer)


def register_grad_sampler() -> None:
    utils.register_grad_sampler(RelativePositionEmbedding)(compute_relative_position_embedding_grad_sample)
