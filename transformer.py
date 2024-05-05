import torch
import torch.nn as nn
from torch.nn import functional as F

n_embedding_dims = 384
n_heads = 6
n_blocks = 6
dropout_ratio = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    def __init__(self, head_size, context_length, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.key = nn.Linear(n_embedding_dims, head_size, bias=False, device=device)
        self.query = nn.Linear(n_embedding_dims, head_size, bias=False, device=device)
        self.value = nn.Linear(n_embedding_dims, head_size, bias=False, device=device)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length, device=device))
        )
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * C**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, context_length, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList(
            [Head(head_size, context_length) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embedding_dims, n_embedding_dims, device=device)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, n_embedding_dims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(n_embedding_dims, n_embedding_dims * 4, device=device),
            nn.ReLU(),
            nn.Linear(n_embedding_dims * 4, n_embedding_dims, device=device),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self, n_embedding_dims, n_heads, context_length, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        head_size = n_embedding_dims // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, context_length)
        self.ffwd = FeedForward(n_embedding_dims)
        self.layer_norm_1 = nn.LayerNorm(n_embedding_dims, device=device)
        self.layer_norm_2 = nn.LayerNorm(n_embedding_dims, device=device)

    def forward(self, x):
        x = x + self.sa(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, context_length, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding_dims, device=device)
        self.position_embedding_table = nn.Embedding(context_length, n_embedding_dims, device=device)
        self.context_length = context_length
        self.blocks = nn.Sequential(
            *[
                Block(n_embedding_dims, n_heads=n_heads, context_length=context_length)
                for _ in range(n_blocks)
            ],
            nn.LayerNorm(n_embedding_dims, device=device),
        )
        self.lm_head = nn.Linear(n_embedding_dims, vocab_size, device=device)
        self.ffwd = FeedForward(n_embedding_dims)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device))
        token_embeddings = self.token_embedding_table(inputs)
        x = position_embeddings + token_embeddings
        x = self.blocks(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens, streamer):
        for _ in range(max_new_tokens):
            current_inputs = inputs[:, -self.context_length :]
            logits, _loss = self(current_inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            if streamer:
                streamer(next_token[0].tolist())
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs
