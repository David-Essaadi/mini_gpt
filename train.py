import torch
from pathlib import Path
from transformer import Transformer
from tokenizer import Tokenizer


with open("input.txt", "r", encoding="ascii") as f:
    text = f.read()

tokenizer = Tokenizer(sorted(set(text)))

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
validation_data = data[n:]

# Hyperparameters
context_length = 256
learning_rate = 3e-4
batch_size = 64
max_steps = 5000
eval_interval = 100
eval_iters = 5


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


def get_batch(split):
    data = train_data if split == "train" else validation_data
    random_indexes = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in random_indexes])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in random_indexes])
    return x, y


m = Transformer(tokenizer.vocab_size, context_length)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for step in range(max_steps):
    print(step)
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    inputs, targets = get_batch("train")
    logits, loss = m(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


Path("output").mkdir(exist_ok=True)
torch.save(m.state_dict(), "output/model.pt")
print("done saving")

# Needs to be set to disable dropout and norm layers
m.eval()
print(tokenizer.decode(m.generate(inputs, 10)[0].tolist()))
