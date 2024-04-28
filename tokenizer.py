class Tokenizer:
    def __init__(self, vocab) -> None:
        self.vocab_size = len(vocab)
        self.stoi = {char: i for i, char in enumerate(vocab)}
        self.itos = {i: char for i, char in enumerate(vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, output_ids: list[int]) -> str:
        return "".join([self.itos[id] for id in output_ids])
