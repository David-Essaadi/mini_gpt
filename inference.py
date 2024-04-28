import torch

from tokenizer import Tokenizer
from transformer import Transformer

with open("input.txt", "r", encoding="ascii") as f:
    text = f.read()

chars = sorted(set(text))
tokenizer = Tokenizer(sorted(set(text)))

model = Transformer(tokenizer.vocab_size, 256)
model.load_state_dict(torch.load("output/model.pt"))
model.eval()
inputs = torch.zeros((1, 1), dtype=torch.long)
print(tokenizer.decode(model.generate(inputs, 400)[0].tolist()))
