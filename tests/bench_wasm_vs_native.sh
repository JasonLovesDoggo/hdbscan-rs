#!/bin/bash
# Benchmark: WASM (Node.js) vs Native (Rust release)
# Requires: rustup target wasm32-unknown-unknown, wasm-bindgen-cli, node
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WASM_DIR=$(mktemp -d)
trap 'rm -rf "$WASM_DIR"' EXIT

echo "Building native release..."
cargo build --release --example bench_wasm_compare --manifest-path "$REPO_DIR/Cargo.toml" 2>/dev/null

echo "Building WASM release..."
cargo build --target wasm32-unknown-unknown --features wasm --no-default-features --profile wasm-release --manifest-path "$REPO_DIR/Cargo.toml" 2>/dev/null

echo "Generating WASM bindings..."
wasm-bindgen --target nodejs --out-dir "$WASM_DIR" "$REPO_DIR/target/wasm32-unknown-unknown/wasm-release/hdbscan_rs.wasm"

# Optimize if wasm-opt is available
if command -v wasm-opt &>/dev/null || npx wasm-opt --version &>/dev/null 2>&1; then
    echo "Optimizing WASM with wasm-opt..."
    npx wasm-opt -Os --enable-bulk-memory "$WASM_DIR/hdbscan_rs_bg.wasm" -o "$WASM_DIR/hdbscan_rs_bg.wasm" 2>/dev/null || true
fi

cp "$SCRIPT_DIR/bench_wasm.mjs" "$WASM_DIR/bench.mjs"

WASM_SIZE=$(ls -lh "$WASM_DIR/hdbscan_rs_bg.wasm" | awk '{print $5}')

echo ""
echo "WASM binary size: $WASM_SIZE"
echo ""

echo "--- Running Native ---"
"$REPO_DIR/target/release/examples/bench_wasm_compare"
echo ""
echo "--- Running WASM (Node.js) ---"
node "$WASM_DIR/bench.mjs"
