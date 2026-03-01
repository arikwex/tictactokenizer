const fs = require("fs");
const path = require("path");

function inlineBinaryWeights() {
  return {
    name: "inline-binary-weights",
    load(id) {
      if (!id.endsWith(".pt")) return null;
      const absPath = path.resolve(id);
      const bytes = fs.readFileSync(absPath);
      const base64 = bytes.toString("base64");
      const code = `
        const base64 = "${base64}";
        let binaryString;
        if (typeof atob === "function") {
          binaryString = atob(base64);
        } else {
          binaryString = Buffer.from(base64, "base64").toString("binary");
        }
        const data = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          data[i] = binaryString.charCodeAt(i);
        }
        export default data;
      `;
      return code;
    },
  };
}

function inlineJsonConfig() {
  return {
    name: "inline-json-config",
    load(id) {
      if (!id.endsWith(".json")) return null;
      const absPath = path.resolve(id);
      const raw = fs.readFileSync(absPath, "utf8");
      const parsed = JSON.parse(raw);
      const code = `export default ${JSON.stringify(parsed)};`;
      return { code, map: { mappings: "" } };
    },
  };
}

function singleFileHtml(options = {}) {
  const {
    title = "TicTacTokenizer",
    description = "Play Tic-Tac-Toe against a tiny Transformer model running entirely in your browser.",
    inlineCss = "",
  } = options;

  return {
    name: "single-file-html",
    generateBundle(_, bundle) {
      const chunks = Object.values(bundle).filter((item) => item.type === "chunk");
      if (chunks.length !== 1) {
        this.error("Expected exactly one JavaScript chunk to inline.");
      }
      const chunk = chunks[0];
      delete bundle[chunk.fileName];
      const escapedCode = chunk.code.split("</script>").join("<\\/script>");
      const scriptTag = `<script>${escapedCode}</script>`;
      const html = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        `  <title>${title}</title>`,
        `  <meta name="description" content="${description}" />`,
        inlineCss ? `  <style>${inlineCss}</style>` : "",
        "</head>",
        "<body>",
        '  <div id="app"></div>',
        `  ${scriptTag}`,
        "</body>",
        "</html>",
      ]
        .filter(Boolean)
        .join("\n");

      this.emitFile({
        type: "asset",
        fileName: "index.html",
        source: html,
      });
    },
  };
}

const baseCss = `
  :root {
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #0f172a;
    background: #f8fafc;
  }
  * {
    box-sizing: border-box;
  }
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #app {
    width: min(480px, calc(100vw - 2rem));
    padding: 1.5rem;
    border-radius: 1rem;
    background: white;
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.15);
  }
  h1 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    font-size: 1.75rem;
    text-align: center;
  }
  p.subtitle {
    margin-top: 0;
    margin-bottom: 1.5rem;
    text-align: center;
    color: #475569;
  }
  .board {
    width: 100%;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  .cell {
    aspect-ratio: 1 / 1;
    border: 2px solid #e2e8f0;
    border-radius: 0.75rem;
    font-size: 1.75rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    color: #0f172a;
    transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    background: #f8fafc;
  }
  .cell:hover {
    transform: translateY(-2px);
    border-color: #94a3b8;
  }
  .cell.disabled {
    cursor: not-allowed;
    opacity: 0.6;
    transform: none;
  }
  .cell.x {
    color: #dc2626;
  }
  .cell.o {
    color: #0891b2;
  }
  .status {
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 600;
  }
  .token-bar {
    font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
    font-size: 0.85rem;
    background: #f1f5f9;
    border-radius: 0.5rem;
    padding: 0.5rem;
    text-align: center;
    margin-bottom: 1rem;
    white-space: nowrap;
    overflow-x: auto;
  }
  button.new-game {
    width: 100%;
    border: none;
    border-radius: 999px;
    background: #0ea5e9;
    color: white;
    font-size: 1rem;
    padding: 0.75rem 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.35);
  }
  button.new-game:hover {
    background: #0284c7;
    transform: translateY(-1px);
  }
  button.new-game:active {
    transform: translateY(0);
  }
`;

module.exports = {
  input: "tictactokenizer_game.js",
  output: {
    file: "public/bundle.js",
    format: "iife",
    sourcemap: false,
  },
  plugins: [
    inlineBinaryWeights(),
    inlineJsonConfig(),
    singleFileHtml({ inlineCss: baseCss }),
  ],
};
