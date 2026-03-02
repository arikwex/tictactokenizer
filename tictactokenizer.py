"""
Train a tiny GPT to play tic-tac-toe moves using an on-the-fly game engine and a bespoke grammar.
Every training sample follows the sequence: BOS, [9 board cells], MOV, <move token>.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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
introspection_board = ["_", "X", "O", "_", "X", "_", "_", "_", "_"]
# introspection_board = ["_", "_", "_", "_", "X", "_", "X", "O", "_"]

# Hyperparameters loaded from shared config
MODEL_CONFIG_PATH = "model_config.json"
with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
    model_config = json.load(cfg_file)

n_layer = model_config["n_layer"]
n_embd = model_config["n_embd"]
block_size = model_config["block_size"]  # BOS + 9 board cells + MOV
n_head = model_config["n_head"]

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
num_steps = 4000
batch_size = 128
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
    
    def select_best_move(self, board: List[str]) -> int:
        legal = self.legal_moves(board)
        if not legal:
            return -1
        player = self.current_player(board)
        opponent = "O" if player == "X" else "X"

        # Prefer to win on the next move
        winning = []
        for move in legal:
            temp = board.copy()
            temp[move] = player
            if self.check_winner(temp) == player:
                winning.append(move)
        if winning:
            return random.choice(winning)

        # Prevent the opponent from winning on the next move
        blocking = []
        for move in legal:
            temp = board.copy()
            temp[move] = opponent
            if self.check_winner(temp) == opponent:
                blocking.append(move)
        if blocking:
            return random.choice(blocking)

        # If the center is empty, return it
        if board[4] == "_":
            return 4
        
        #  Otherwise random valid move
        return random.choice(legal)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
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
        for idx, block in enumerate(self.blocks):
            x = block(x)
        logits = self.lm_head(x)
        return logits

    def forward_with_activations(self, idx: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.block_size, "sequence length exceeds block size"
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.input_norm(x)
        activations = [x.detach().cpu()]
        for idx, block in enumerate(self.blocks):
            x = block(x)
            activations.append(x.detach().cpu())
        logits = F.softmax(self.lm_head(x), dim=-1) * 2 - 1
        return logits, activations


def generate_training_sequence(
    engine: TicTacToeEngine
) -> List[int] | Tuple[List[int], List[str], int]:
    board: List[str]
    while True:
        board = ["_"] * 9
        # if random.random() < 0.5:
        #     board = introspection_board.copy()
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

    # if engine.pretty(board) == engine.pretty(introspection_board):
    #     print("!!!!!!! TRAINING ON INTROSPECTION BOARD !!!!!!!")
    move = engine.select_best_move(board)
    seq = [BOS_ID] + [stoi[cell] for cell in board] + [MOV_ID, stoi[str(move + 1)]]
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
        # seq = seq[: block_size + 1]
        seq_len = len(seq) - 1
        x[i, :seq_len] = torch.tensor(seq[:-1], dtype=torch.long)
        y[i, :seq_len] = torch.tensor(seq[1:], dtype=torch.long)
        # Only train on predicting the move immediately after the MOV token.
        # prefix = seq[:-1]
        # try:
        #     mov_idx = prefix.index(MOV_ID)
        # except ValueError as exc:  # pragma: no cover - should never happen
        #     raise RuntimeError("training sequence missing MOV token") from exc
        # mask[i, mov_idx] = 1.0
        mask[i, -1] = 1.0
    return x.to(device), y.to(device), mask.to(device)


def _value_to_color(val: float) -> Tuple[int, int, int]:
    clipped = max(-1.0, min(1.0, float(val)))
    t = (clipped + 1.0) * 0.5  # map [-1, 1] -> [0, 1]
    start = (6, 16, 64)  # dark blue
    end = (0, 255, 128)  # bright green
    r = int(round(start[0] + (end[0] - start[0]) * t))
    g = int(round(start[1] + (end[1] - start[1]) * t))
    b = int(round(start[2] + (end[2] - start[2]) * t))
    return (r, g, b)


def render_activation_grid(
    activations: List[torch.Tensor],
    token_ids: List[int],
    path: str,
    board: List[str] | None = None,
    board_activations: torch.Tensor | None = None,
) -> None:
    stage_labels = ["input"] + [f"block {i + 1}" for i in range(len(activations) - 1)]
    mats = [act.squeeze(0).cpu() for act in activations]
    token_count = mats[0].shape[0]
    embed_dim = mats[0].shape[1]
    assert len(token_ids) == token_count, "token count mismatch"

    border = 20
    stage_spacing = 30
    token_spacing = 14
    dim_height = 6
    cell_width = 18
    label_height = 18
    row_label_width = 50

    cell_height = embed_dim * dim_height
    content_top = border + label_height + 8
    content_left = border + row_label_width
    board_column_width = 150 if board is not None else 0
    activation_width = len(mats) * cell_width + (len(mats) - 1) * stage_spacing
    board_spacing = stage_spacing if board is not None else 0
    img_width = content_left + activation_width + board_spacing + board_column_width + border
    img_height = content_top + token_count * cell_height + (token_count - 1) * token_spacing + border
    img = Image.new("RGB", (img_width, img_height), color="#05060a")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Stage labels
    def measure(text: str) -> Tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    for idx_stage, label in enumerate(stage_labels):
        x = content_left + idx_stage * (cell_width + stage_spacing)
        text_w, text_h = measure(label)
        draw.text(
            (x + (cell_width - text_w) / 2, border + (label_height - text_h) / 2),
            label,
            fill="#f8fafc",
            font=font,
        )

    token_labels = [itos[token_id] for token_id in token_ids]
    for token_idx, label in enumerate(token_labels):
        y = content_top + token_idx * (cell_height + token_spacing)
        text_w, text_h = measure(label)
        draw.text(
            (border + row_label_width - text_w - 6, y + (cell_height - text_h) / 2),
            label,
            fill="#e2e8f0",
            font=font,
        )

    for stage_idx, stage_mat in enumerate(mats):
        x0 = content_left + stage_idx * (cell_width + stage_spacing)
        for token_idx in range(token_count):
            y0 = content_top + token_idx * (cell_height + token_spacing)
            vec = stage_mat[token_idx]
            for dim_idx in range(embed_dim):
                val = vec[dim_idx].item()
                color = _value_to_color(val)
                y_start = y0 + dim_idx * dim_height
                y_end = y_start + dim_height - 1
                draw.rectangle(
                    [x0, y_start, x0 + cell_width - 1, y_end],
                    fill=color,
                )

    if board is not None and board_activations is not None and board_column_width > 0:
        cell_size = board_column_width // 3
        board_height = cell_size * 3
        x_board = content_left + activation_width + board_spacing
        y_board = img_height - border - board_height
        board_tensor = board_activations.squeeze(0)[-1]
        board_colors: List[Tuple[int, int, int]] = []
        argmax_idx = -1
        max_val = -float("inf")
        print(board_tensor)
        for idx_cell in range(len(board)):
            token_idx = 2 + idx_cell
            val = torch.tanh(board_tensor[token_idx]).item()
            if val > max_val:
                max_val = val
                argmax_idx = idx_cell
            board_colors.append(_value_to_color(val))
        # board_colors[argmax_idx] = (255, 0, 0)
        for row in range(3):
            for col in range(3):
                idx_cell = row * 3 + col
                cell_x = x_board + col * cell_size
                cell_y = y_board + row * cell_size
                fill_color = board_colors[idx_cell]
                draw.rectangle(
                    [cell_x, cell_y, cell_x + cell_size - 6, cell_y + cell_size - 6],
                    fill=fill_color,
                    outline="#0f172a",
                    width=3,
                )
                token_char = board[idx_cell]
                if token_char != "_":
                    text_w, text_h = measure(token_char)
                    draw.text(
                        (cell_x + (cell_size - text_w) / 2, cell_y + (cell_size - text_h) / 2),
                        token_char,
                        fill="#f8fafc",
                        font=font,
                    )

    img.save(path)


def introspect_model(model: MicroGPT, engine: TicTacToeEngine, device: torch.device, board: List[str], path: str = "current_introspection.png") -> None:
    model.eval()
    context_tokens = [BOS_ID] + [stoi[cell] for cell in board] + [MOV_ID]
    idx = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        board_activations, activations = model.forward_with_activations(idx)
    render_activation_grid(activations, context_tokens, path, board=board, board_activations=board_activations)
    human_board = engine.pretty(board, colored=False)
    print("Generated introspection sample:")
    print(human_board)
    print(f"Saved activation visualization to {path}")


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


def interactive_loop(model: MicroGPT, device: torch.device, introspect: bool = False) -> None:
    engine = TicTacToeEngine()
    print("\nInteractive play started. You are X and move first. Ctrl+C to exit.\n")
    while True:
        board = ["_"] * 9
        print(engine.pretty(board))
        while True:
            if introspect:
                introspect_model(model, engine, device, board)
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


def run_training(
    model: MicroGPT,
    engine: TicTacToeEngine,
    device: torch.device,
    introspect: bool = False,
) -> None:
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

        if introspect and (step + 1) % 100 == 0:
            introspect_model(model, engine, device, introspection_board)

    save_quantized_weights(model, weights_path)
    print(f"\nSaved quantized weights to {weights_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and interact with the TicTacTokenizer micro transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model from scratch and overwrite the quantized weights file.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Launch the interactive CLI game against the model.",
    )
    parser.add_argument(
        "--introspect",
        action="store_true",
        help="Render a PNG visualization of activations across transformer blocks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.train or args.play or args.introspect):
        print("No action specified. Use one or more of --train, --play, --introspect.")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    engine = TicTacToeEngine()
    model = MicroGPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    expected_bytes = _param_count(model)

    weights_exist = os.path.exists(weights_path) and os.path.getsize(weights_path) == expected_bytes

    if args.train or not weights_exist:
        if not args.train and not weights_exist:
            print("Weights missing or invalid size; training a new model.")
        run_training(model, engine, device, introspect=args.introspect)
    else:
        load_quantized_weights(model, weights_path)
        model.to(device)
        print(f"Loaded weights from {weights_path}")

    if args.introspect and not args.play:
        introspect_model(model, engine, device, introspection_board)
    if args.play:
        interactive_loop(model, device, introspect=args.introspect)


if __name__ == "__main__":
    main()
    # engine_test = TicTacToeEngine()
    # engine_test.apply_move(1)
    # engine_test.apply_move(2)
    # engine_test.apply_move(4)
    # print(engine_test.pretty() + "\n")
    # print(engine_test.legal_moves())
    # print(engine_test.select_best_move(engine_test.board))
