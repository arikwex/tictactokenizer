"""
Train a tiny GPT to play tic-tac-toe moves using an on-the-fly game engine and a bespoke grammar.
Every training sample follows the sequence: BOS, [9 board cells], MOV, <move token>.
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

RED = "\033[31m"
TEAL = "\033[36m"
GRAY = "\033[90m"
RESET = "\033[0m"

# Grammar tokens: BOS, MOV, digits 1-9 (move positions), and board characters _, X, O.
TOKENS = ["BOS", "MOV"] + [str(i) for i in range(1, 10)] + ["_", "X", "O"]
stoi = {tok: idx for idx, tok in enumerate(TOKENS)}
itos = {idx: tok for tok, idx in stoi.items()}
BOS_ID = stoi["BOS"]
MOV_ID = stoi["MOV"]
MOVE_TOKEN_IDS = [stoi[str(i)] for i in range(1, 10)]
vocab_size = len(TOKENS)
print(f"vocab size: {vocab_size}")

# Hyperparameters
n_layer = 2
n_embd = 12
block_size = 12  # BOS + 9 board cells + MOV
n_head = 2
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
num_steps = 2000
batch_size = 64
weights_path = "tictactokenizer_weights.pt"


class TicTacToeEngine:
    WIN_PATTERNS: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )

    def __init__(self) -> None:
        self.board: List[str] = ["_"] * 9

    def reset(self) -> None:
        self.board = ["_"] * 9

    def current_player(self, board: List[str] | None = None) -> str:
        board_ref = self.board if board is None else board
        x_count = board_ref.count("X")
        o_count = board_ref.count("O")
        return "X" if x_count == o_count else "O"

    def legal_moves(self, board: List[str] | None = None) -> List[int]:
        board_ref = self.board if board is None else board
        return [idx for idx, cell in enumerate(board_ref) if cell == "_"]

    def apply_move(self, move: int) -> str:
        if self.board[move] != "_":
            raise ValueError(f"cell {move + 1} is already occupied")
        player = self.current_player(self.board)
        self.board[move] = player
        return player

    def check_winner(self, board: List[str] | None = None) -> str | None:
        board_ref = self.board if board is None else board
        for a, b, c in self.WIN_PATTERNS:
            if board_ref[a] != "_" and board_ref[a] == board_ref[b] == board_ref[c]:
                return board_ref[a]
        if "_" not in board_ref:
            return "draw"
        return None

    def random_state_and_move(self) -> Tuple[List[str], int]:
        _, board, move = generate_training_sequence(self, return_state=True)
        return board, move

    def pretty(self, board: List[str] | None = None, show_indices: bool = False, colored: bool = True) -> str:
        board_ref = self.board if board is None else board

        def format_cell(cell: str, idx: int) -> str:
            if cell == "_" and show_indices:
                return f"{GRAY}{idx + 1}{RESET}" if colored else str(idx + 1)
            if cell == "X":
                return f"{RED}X{RESET}" if colored else "X"
            if cell == "O":
                return f"{TEAL}O{RESET}" if colored else "O"
            return "_" if cell == "_" else cell

        rows = []
        for row_start in range(0, 9, 3):
            row_cells = [format_cell(board_ref[idx], idx) for idx in range(row_start, row_start + 3)]
            rows.append(" | ".join(row_cells))
        return "\n---------\n".join(rows)


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


def generate_training_sequence(
    engine: TicTacToeEngine, return_state: bool = False
) -> List[int] | Tuple[List[int], List[str], int]:
    def select_best_move(board: List[str]) -> int:
        legal = engine.legal_moves(board)
        if not legal:
            return -1
        player = engine.current_player(board)
        opponent = "O" if player == "X" else "X"

        # Prefer to win on the next move
        winning = []
        for move in legal:
            temp = board.copy()
            temp[move] = player
            if engine.check_winner(temp) == player:
                winning.append(move)
        if winning:
            return random.choice(winning)

        # Prevent the opponent from winning on the next move
        blocking = []
        for move in legal:
            temp = board.copy()
            temp[move] = opponent
            if engine.check_winner(temp) == opponent:
                blocking.append(move)
        if blocking:
            return random.choice(blocking)

        # If the center is empty, return it
        if board[4] == "_":
            return 4
        
        #  Otherwise random valid move
        return random.choice(legal)

    board: List[str]
    while True:
        board = ["_"] * 9
        max_moves = random.randint(0, 8)
        for _ in range(max_moves):
            legal = engine.legal_moves(board)
            if not legal:
                break
            move = random.choice(legal)
            board[move] = engine.current_player(board)
            if engine.check_winner(board) is not None:
                break
        if engine.check_winner(board) is None and engine.legal_moves(board):
            break

    move = select_best_move(board)
    seq = [BOS_ID] + [stoi[cell] for cell in board] + [MOV_ID, stoi[str(move + 1)]]
    if return_state:
        return seq, board.copy(), move
    return seq


def sample_batch(
    batch_size: int,
    block_size: int,
    engine: TicTacToeEngine,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.full((batch_size, block_size), BOS_ID, dtype=torch.long)
    y = torch.full((batch_size, block_size), BOS_ID, dtype=torch.long)
    mask = torch.zeros((batch_size, block_size), dtype=torch.float32)
    for i in range(batch_size):
        seq = generate_training_sequence(engine)
        seq = seq[: block_size + 1]
        seq_len = len(seq) - 1
        x[i, :seq_len] = torch.tensor(seq[:-1], dtype=torch.long)
        y[i, :seq_len] = torch.tensor(seq[1:], dtype=torch.long)
        mask[i, :seq_len] = 1.0
    return x.to(device), y.to(device), mask.to(device)


def preview_model(model: MicroGPT, engine: TicTacToeEngine, device: torch.device, samples: int = 5) -> None:
    print("\n--- inference (tic-tac-toe move previews) ---")
    model.eval()
    with torch.no_grad():
        for sample_idx in range(samples):
            board, _ = engine.random_state_and_move()
            context = torch.tensor(
                [[BOS_ID] + [stoi[cell] for cell in board] + [MOV_ID]],
                dtype=torch.long,
                device=device,
            )
            logits = model(context)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            move_probs = probs[:, MOVE_TOKEN_IDS]
            move_choice = torch.argmax(move_probs, dim=-1).item()
            move_token = MOVE_TOKEN_IDS[move_choice]
            move = int(itos[move_token])
            player = engine.current_player(board)
            before = engine.pretty(board)
            after_board = board.copy()
            validity = "valid"
            if after_board[move - 1] == "_":
                after_board[move - 1] = player
            else:
                validity = "invalid (cell already taken)"
            after = engine.pretty(after_board)
            print(f"sample {sample_idx + 1:2d} | predicted move {move} for player {player} -> {validity}")
            print("before:")
            print(before)
            print("after :")
            print(after)
            print("-" * 21)


def model_choose_move(model: MicroGPT, board: List[str], device: torch.device, engine: TicTacToeEngine) -> int:
    legal = engine.legal_moves(board)
    if not legal:
        return -1
    context = torch.tensor(
        [[BOS_ID] + [stoi[cell] for cell in board] + [MOV_ID]],
        dtype=torch.long,
        device=device,
    )
    logits = model(context)[:, -1, :].squeeze(0)
    move_logits = logits[MOVE_TOKEN_IDS].clone()
    for idx, token_id in enumerate(MOVE_TOKEN_IDS):
        cell_idx = int(itos[token_id]) - 1
        if cell_idx not in legal:
            move_logits[idx] = float("-inf")
    if not torch.isfinite(move_logits).any():
        return random.choice(legal) + 1
    move_idx = torch.argmax(move_logits).item()
    return int(itos[MOVE_TOKEN_IDS[move_idx]])


def interactive_loop(model: MicroGPT, device: torch.device) -> None:
    engine = TicTacToeEngine()
    print("\nInteractive play started. You are X and move first. Ctrl+C to exit.\n")
    while True:
        board = ["_"] * 9
        print(engine.pretty(board))
        while True:
            print("\nYour move (enter 1-9 for the positions shown below):")
            print(engine.pretty(board, show_indices=True))
            try:
                user_input = input("> ").strip()
            except EOFError:
                print("\nEOF received. Exiting interactive session.")
                return
            if not user_input.isdigit():
                print("Please enter a digit between 1 and 9.")
                continue
            move = int(user_input)
            if move < 1 or move > 9:
                print("Move must be between 1 and 9.")
                continue
            if board[move - 1] != "_":
                print("Cell already occupied. Choose another move.")
                continue
            board[move - 1] = "X"
            print("\nYou played:")
            print(engine.pretty(board))
            winner = engine.check_winner(board)
            if winner or "_" not in board:
                result = "draw" if winner == "draw" or winner is None else f"{winner} wins!"
                print(f"Game over: {result}")
                break

            ai_player = engine.current_player(board)
            ai_move = model_choose_move(model, board, device, engine)
            if ai_move == -1:
                print("Model had no legal moves. Declaring draw.")
                break
            board[ai_move - 1] = ai_player
            print(f"\nModel plays {ai_move} as {ai_player}:")
            print(engine.pretty(board))
            winner = engine.check_winner(board)
            if winner or "_" not in board:
                result = "draw" if winner == "draw" or winner is None else f"{winner} wins!"
                print(f"Game over: {result}")
                break

        print("\nStarting a new game...\n")


def _flatten_state_dict(model: MicroGPT) -> torch.Tensor:
    flats = [tensor.detach().cpu().float().reshape(-1) for tensor in model.state_dict().values()]
    return torch.cat(flats)


def _param_count(model: MicroGPT) -> int:
    return sum(tensor.numel() for tensor in model.state_dict().values())


def save_quantized_weights(model: MicroGPT, path: str) -> None:
    flat = _flatten_state_dict(model)
    clamped = torch.clamp(flat, -1.0, 1.0)
    quantized = torch.round(clamped * 127).to(torch.int16)
    uint8_vals = torch.where(quantized < 0, quantized + 256, quantized).to(torch.uint8)
    data = bytes(uint8_vals.tolist())
    with open(path, "wb") as f:
        f.write(data)


def load_quantized_weights(model: MicroGPT, path: str) -> None:
    with open(path, "rb") as f:
        raw = f.read()
    uint = torch.tensor(list(raw), dtype=torch.int16)
    signed = torch.where(uint >= 128, uint - 256, uint)
    dequant = torch.clamp(signed.float() / 127.0, -1.0, 1.0)
    pointer = 0
    new_state = {}
    state_dict = model.state_dict()
    for key, tensor in state_dict.items():
        numel = tensor.numel()
        chunk = dequant[pointer : pointer + numel]
        if chunk.numel() != numel:
            raise ValueError("Quantized weight file is incompatible with the current model architecture.")
        pointer += numel
        new_state[key] = chunk.view_as(tensor).to(tensor.dtype)
    if pointer != dequant.numel():
        raise ValueError("Quantized weight file has extra data.")
    model.load_state_dict(new_state)


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    engine = TicTacToeEngine()
    model = MicroGPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    expected_bytes = _param_count(model)

    if os.path.exists(weights_path) and os.path.getsize(weights_path) == expected_bytes:
        load_quantized_weights(model, weights_path)
        model.to(device)
        print(f"Loaded weights from {weights_path}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps_adam)

        for step in range(num_steps):
            model.train()
            xb, yb, mask = sample_batch(batch_size, block_size, engine, device)
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

        save_quantized_weights(model, weights_path)
        print(f"\nSaved quantized weights to {weights_path}")

    preview_model(model, engine, device)
    interactive_loop(model, device)


if __name__ == "__main__":
    main()
