import torch

from tokenizer import Tokenizer
from transformer import Transformer

with open("input.txt", "r", encoding="ascii") as f:
    text = f.read()

chars = sorted(set(text))
tokenizer = Tokenizer(sorted(set(text)))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer(tokenizer.vocab_size, 256)
model.load_state_dict(torch.load("output/model.pt"))
model.eval()

def streamer(tokens):
    print(tokenizer.decode(tokens), end="")

inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
while True:
    model.generate(inputs, 1000, streamer)
