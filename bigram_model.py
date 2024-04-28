import torch
import torch.nn as nn
from torch.nn import functional as F

n_embedding_dims = 32


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding_dims)
        self.position_embedding_table = nn.Embedding(context_length, n_embedding_dims)
        self.lm_head = nn.Linear(n_embedding_dims, vocab_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        position_embeddings = self.position_embedding_table(torch.arange(T))
        token_embeddings = self.token_embedding_table(inputs)
        x = position_embeddings + token_embeddings
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _loss = self(inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs
