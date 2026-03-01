"""
Pytorch implementation of microgpt (just for training performance)
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Dataset: load names just like the original script
if not os.path.exists("input.txt"):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")

docs = [line.strip() for line in open("input.txt") if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer
uchars = sorted(set("".join(docs)))
stoi = {ch: i for i, ch in enumerate(uchars)}
itos = {i: ch for ch, i in stoi.items()}
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

docs_tokens = [[BOS] + [stoi[ch] for ch in doc] + [BOS] for doc in docs]

# Hyperparameters shared with the pure-python version
n_layer = 2
n_embd = 16
block_size = 64
n_head = 4
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
num_steps = 1000
batch_size = 32
temperature = 0.5


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "embedding dimension must be divisible by number of heads"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
            1, 1, block_size, block_size
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(attn, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class GPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.mlp_norm = RMSNorm()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.fc2(F.relu(self.fc1(self.mlp_norm(x))))
        return x


class MicroGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.input_norm = RMSNorm()
        self.blocks = nn.ModuleList(
            [GPTBlock(n_embd=n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, "sequence length exceeds block size"
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits


def sample_batch(
    batch_size: int, block_size: int, docs_tokens: List[List[int]], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.full((batch_size, block_size), BOS, dtype=torch.long)
    y = torch.full((batch_size, block_size), BOS, dtype=torch.long)
    mask = torch.zeros((batch_size, block_size), dtype=torch.float32)
    for i in range(batch_size):
        tokens = random.choice(docs_tokens)
        seq = tokens[: block_size + 1]
        seq_len = len(seq) - 1
        x[i, :seq_len] = torch.tensor(seq[:-1], dtype=torch.long)
        y[i, :seq_len] = torch.tensor(seq[1:], dtype=torch.long)
        mask[i, :seq_len] = 1.0
    return x.to(device), y.to(device), mask.to(device)


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = MicroGPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps_adam)

    for step in range(num_steps):
        model.train()
        xb, yb, mask = sample_batch(batch_size, block_size, docs_tokens, device)
        logits = model(xb)
        loss_all = F.cross_entropy(
            logits.view(-1, vocab_size),
            yb.view(-1),
            reduction="none",
        )
        loss = (loss_all * mask.view(-1)).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        lr_t = learning_rate * (1 - step / num_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr_t
        optimizer.step()
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end="\r")

    print("\n--- inference (new, hallucinated names) ---")
    model.eval()
    with torch.no_grad():
        for sample_idx in range(20):
            tokens = torch.tensor([[BOS, stoi["s"], stoi["t"]]], dtype=torch.long, device=device)
            sample = ['s', 't']
            for _ in range(block_size):
                context = tokens[:, -block_size:]
                logits = model(context)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1)
                next_id = token_id.item()
                if next_id == BOS:
                    break
                sample.append(itos[next_id])
                tokens = torch.cat([tokens, token_id], dim=1)
            print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")


if __name__ == "__main__":
    main()
