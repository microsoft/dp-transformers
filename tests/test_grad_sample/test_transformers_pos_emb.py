import torch

from transformers.models.bart.modeling_bart import  BartLearnedPositionalEmbedding

from opacus.tests.grad_samples.common import GradSampleHooks_test

from dp_transformers.grad_sample.transformers.positional_embedding import register_grad_sampler


class TestPositionalEmbedding(GradSampleHooks_test):
    def test_grad_sample(self):
        """
        Verify that our custom implementation of the grad sampler for huggingface's
        BartLearnedPositionalEmbedding layer works.
        We largely build on the test routines in opacus's library.
        """
        register_grad_sampler()
        batch_size = 1
        max_pos_embs = 10
        embed_dim = 3
    
        x = torch.randint(0, max_pos_embs - 1, (batch_size, embed_dim))
        layer = BartLearnedPositionalEmbedding(max_pos_embs, embed_dim)
        self.run_test(x, layer, batch_first=True)


if __name__ == "__main__":
    test = TestPositionalEmbedding()
    test.test_grad_sample()