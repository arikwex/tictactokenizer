import weightBytes from "./tictactokenizer_weights.pt";
import modelConfig from "./model_config.json";

const TOKENS = ["BOS", "MOV"]
  .concat(Array.from({ length: 9 }, (_, i) => String(i + 1)))
  .concat(["_", "X", "O"]);
const stoi = Object.fromEntries(TOKENS.map((tok, idx) => [tok, idx]));
const MOVE_TOKEN_IDS = Array.from({ length: 9 }, (_, i) => stoi[String(i + 1)]);
const BOS_ID = stoi["BOS"];
const MOV_ID = stoi["MOV"];

const vocabSize = TOKENS.length;
const nLayer = modelConfig.n_layer;
const nEmb = modelConfig.n_embd;
const blockSize = modelConfig.block_size;
const nHead = modelConfig.n_head;
const headDim = nEmb / nHead;
const weightSizeKB = (weightBytes.length / 1024).toFixed(1);

const WIN_PATTERNS = [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [0, 3, 6],
  [1, 4, 7],
  [2, 5, 8],
  [0, 4, 8],
  [2, 4, 6],
];

function loadQuantizedWeights(bytes) {
  const values = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    const byte = bytes[i];
    const signed = byte >= 128 ? byte - 256 : byte;
    values[i] = Math.max(-1, Math.min(1, signed / 127));
  }
  let offset = 0;
  const takeMatrix = (rows, cols) => {
    const mat = Array.from({ length: rows }, () => new Array(cols));
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        mat[r][c] = values[offset++];
      }
    }
    return mat;
  };
  const params = {};
  params.tokenEmb = takeMatrix(vocabSize, nEmb);
  params.posEmb = takeMatrix(blockSize, nEmb);
  params.blocks = [];
  for (let layer = 0; layer < nLayer; layer++) {
    params.blocks.push({
      wq: takeMatrix(nEmb, nEmb),
      wk: takeMatrix(nEmb, nEmb),
      wv: takeMatrix(nEmb, nEmb),
      wo: takeMatrix(nEmb, nEmb),
      fc1: takeMatrix(4 * nEmb, nEmb),
      fc2: takeMatrix(nEmb, 4 * nEmb),
    });
  }
  params.lmHead = takeMatrix(vocabSize, nEmb);
  if (offset !== values.length) {
    throw new Error("Weight file contained unexpected length.");
  }
  return params;
}

function rmsnorm(vec) {
  let meanSq = 0;
  for (let i = 0; i < vec.length; i++) {
    meanSq += vec[i] * vec[i];
  }
  meanSq /= vec.length;
  const scale = 1 / Math.sqrt(meanSq + 1e-5);
  return vec.map((v) => v * scale);
}

function linear(vec, weight) {
  const out = new Array(weight.length);
  for (let i = 0; i < weight.length; i++) {
    let sum = 0;
    const row = weight[i];
    for (let j = 0; j < vec.length; j++) {
      sum += row[j] * vec[j];
    }
    out[i] = sum;
  }
  return out;
}

function relu(vec) {
  return vec.map((v) => (v > 0 ? v : 0));
}

function softmax(logits) {
  let maxVal = -Infinity;
  for (const val of logits) {
    if (val > maxVal) maxVal = val;
  }
  const exps = logits.map((val) =>
    val === -Infinity ? 0 : Math.exp(val - maxVal),
  );
  const sum = exps.reduce((acc, v) => acc + v, 0);
  if (sum === 0) {
    const len = logits.length;
    return logits.map(() => 1 / len);
  }
  return exps.map((val) => val / sum);
}

function cloneMatrix(mat) {
  return mat.map((row) => row.slice());
}

function forward(params, idx) {
  const T = idx.length;
  if (T > blockSize) {
    throw new Error("Sequence length exceeds block size");
  }
  let x = Array.from({ length: T }, (_, t) => {
    const vec = new Array(nEmb);
    const tokenVec = params.tokenEmb[idx[t]];
    const posVec = params.posEmb[t];
    for (let i = 0; i < nEmb; i++) {
      vec[i] = tokenVec[i] + posVec[i];
    }
    return rmsnorm(vec);
  });

  for (let layer = 0; layer < nLayer; layer++) {
    const block = params.blocks[layer];
    const xResidual = cloneMatrix(x);
    const normed = x.map((vec) => rmsnorm(vec));
    const q = normed.map((vec) => linear(vec, block.wq));
    const k = normed.map((vec) => linear(vec, block.wk));
    const v = normed.map((vec) => linear(vec, block.wv));
    const attnOut = Array.from({ length: T }, () => new Array(nEmb).fill(0));
    const scale = 1 / Math.sqrt(headDim);
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < nHead; h++) {
        const qOffset = h * headDim;
        const scores = new Array(T).fill(-Infinity);
        for (let tp = 0; tp <= t; tp++) {
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += q[t][qOffset + d] * k[tp][qOffset + d];
          }
          scores[tp] = dot * scale;
        }
        const weights = softmax(scores);
        const headVec = new Array(headDim).fill(0);
        for (let tp = 0; tp <= t; tp++) {
          const weight = weights[tp];
          if (weight === 0) continue;
          for (let d = 0; d < headDim; d++) {
            headVec[d] += weight * v[tp][qOffset + d];
          }
        }
        for (let d = 0; d < headDim; d++) {
          attnOut[t][qOffset + d] = headVec[d];
        }
      }
    }
    x = attnOut.map((vec, idxRow) => {
      const proj = linear(vec, block.wo);
      const out = new Array(nEmb);
      for (let i = 0; i < nEmb; i++) {
        out[i] = proj[i] + xResidual[idxRow][i];
      }
      return out;
    });

    const mlpResidual = cloneMatrix(x);
    const mlpNormed = x.map((vec) => rmsnorm(vec));
    const hidden = mlpNormed.map((vec) => relu(linear(vec, block.fc1)));
    const mlpOut = hidden.map((vec) => linear(vec, block.fc2));
    x = mlpOut.map((vec, idxRow) => {
      const out = new Array(nEmb);
      for (let i = 0; i < nEmb; i++) {
        out[i] = vec[i] + mlpResidual[idxRow][i];
      }
      return out;
    });
  }

  const logits = x.map((vec) => linear(vec, params.lmHead));
  return logits;
}

function currentPlayer(board) {
  const xCount = board.filter((c) => c === "X").length;
  const oCount = board.filter((c) => c === "O").length;
  return xCount === oCount ? "X" : "O";
}

function legalMoves(board) {
  const moves = [];
  board.forEach((cell, idx) => {
    if (cell === "_") moves.push(idx);
  });
  return moves;
}

function checkWinner(board) {
  for (const [a, b, c] of WIN_PATTERNS) {
    if (board[a] !== "_" && board[a] === board[b] && board[b] === board[c]) {
      return board[a];
    }
  }
  if (board.every((cell) => cell !== "_")) {
    return "draw";
  }
  return null;
}

function tokensForBoard(board) {
  return [BOS_ID].concat(board.map((cell) => stoi[cell])).concat([MOV_ID]);
}

function modelPickMove(params, board) {
  const moves = legalMoves(board);
  if (moves.length === 0) {
    return -1;
  }
  const contextTokens = tokensForBoard(board);
  const logits = forward(params, contextTokens);
  const lastLogits = logits[logits.length - 1];
  let bestMove = moves[0];
  let bestScore = -Infinity;
  for (const move of moves) {
    const tokenId = MOVE_TOKEN_IDS[move];
    const score = lastLogits[tokenId];
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
  }
  return bestMove;
}

function makeCellLabel(value) {
  return value === "_" ? "" : value;
}

function setupGame(params) {
  const app = document.getElementById("app");
  if (!app) {
    throw new Error("Missing #app container");
  }
  let board = Array(9).fill("_");
  let gameOver = false;
  let thinking = false;

  app.innerHTML = "";

  const title = document.createElement("h1");
  title.textContent = "TicTacTokenizer";

  const subtitle = document.createElement("p");
  subtitle.className = "subtitle";
  subtitle.textContent = "8 Bit Quantized Decoder Transformer";

  const paramGrid = document.createElement("div");
  paramGrid.className = "param-grid";
  const specs = [
    ["n_layer", nLayer],
    ["n_embd", nEmb],
    ["block_size", blockSize],
    ["n_head", nHead],
    ["vocab_size", vocabSize],
    ["weights", `${weightSizeKB} KB`],
  ];
  specs.forEach(([label, value]) => {
    const cell = document.createElement("div");
    const labelEl = document.createElement("span");
    labelEl.className = "label";
    labelEl.textContent = label;
    const valueEl = document.createElement("span");
    valueEl.className = "value";
    valueEl.textContent = String(value);
    cell.appendChild(labelEl);
    cell.appendChild(valueEl);
    paramGrid.appendChild(cell);
  });

  const boardEl = document.createElement("div");
  boardEl.className = "board";

  const tokenBar = document.createElement("div");
  tokenBar.className = "token-bar";

  const statusEl = document.createElement("div");
  statusEl.className = "status";

  const newGameBtn = document.createElement("button");
  newGameBtn.className = "new-game";
  newGameBtn.type = "button";
  newGameBtn.textContent = "New Game";

  const cells = [];
  for (let i = 0; i < 9; i++) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "cell";
    btn.textContent = makeCellLabel(board[i]);
    btn.addEventListener("click", () => {
      if (gameOver || thinking) return;
      if (board[i] !== "_") return;
      board[i] = "X";
      updateBoard();
      maybeFinishTurn();
    });
    boardEl.appendChild(btn);
    cells.push(btn);
  }

  function updateTokens() {
    const tokens = ["BOS"].concat(board).concat(["MOV"]);
    tokenBar.textContent = tokens.map((tok) => `[${tok}]`).join("");
  }

  function updateBoard() {
    cells.forEach((cell, idx) => {
      const value = board[idx];
      cell.textContent = makeCellLabel(value);
      cell.classList.remove("x", "o", "disabled");
      if (value === "X") {
        cell.classList.add("x");
      } else if (value === "O") {
        cell.classList.add("o");
      }
      if (gameOver || thinking || value !== "_") {
        cell.classList.add("disabled");
      }
    });
    updateTokens();
  }

  function setStatus(message) {
    statusEl.textContent = message;
  }

  function resetGame() {
    board = Array(9).fill("_");
    gameOver = false;
    thinking = false;
    setStatus("You are X and move first.");
    cells.forEach((cell, idx) => {
      cell.classList.remove("disabled", "x", "o");
      cell.textContent = makeCellLabel("_");
    });
    updateBoard();
  }

  newGameBtn.addEventListener("click", () => {
    resetGame();
  });

  function conclude(result) {
    gameOver = true;
    thinking = false;
    if (result === "draw") {
      setStatus('Game over: draw. Hit "New Game" to try again.');
    } else if (result === "X") {
      setStatus('You win! Hit "New Game" for a rematch.');
    } else {
      setStatus('Model wins. Hit "New Game" to play again.');
    }
    updateBoard();
  }

  function maybeFinishTurn() {
    const winner = checkWinner(board);
    if (winner) {
      conclude(winner);
      return;
    }
    thinking = true;
    updateBoard();
    setStatus("Model thinking...");
    window.setTimeout(() => {
      const aiMove = modelPickMove(params, board);
      if (aiMove === -1) {
        conclude("draw");
        return;
      }
      const aiPlayer = currentPlayer(board);
      board[aiMove] = aiPlayer;
      const afterWinner = checkWinner(board);
      if (afterWinner) {
        conclude(afterWinner);
      } else {
        thinking = false;
        setStatus("Your turn.");
        cells.forEach((cell, idx) => {
          if (board[idx] === "_") {
            cell.classList.remove("disabled");
            cell.textContent = makeCellLabel("_");
          }
        });
        updateBoard();
      }
    }, 250);
  }

  resetGame();
  app.appendChild(title);
  app.appendChild(subtitle);
  app.appendChild(paramGrid);
  app.appendChild(boardEl);
  app.appendChild(tokenBar);
  app.appendChild(statusEl);
  app.appendChild(newGameBtn);
}

const params = loadQuantizedWeights(weightBytes);

const start = () => setupGame(params);
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", start, { once: true });
} else {
  start();
}
