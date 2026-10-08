#!/usr/bin/env bash
# TypeScript 5.9.3 inline interop bodies, including the ownKeys factory.
set -euo pipefail
fixture_dir="$(cd "$(dirname "$0")" && pwd)"
tool_dir="$(mktemp -d)"
trap 'rm -rf "$tool_dir"' EXIT
npm install --prefix "$tool_dir" --no-audit --no-fund --no-package-lock typescript@5.9.3 terser@5.51.2 >/dev/null
node "$tool_dir/node_modules/typescript/bin/tsc" "$fixture_dir"/src/*.ts \
  --target es5 --module commonjs --esModuleInterop --outDir "$fixture_dir/generated"

NODE_PATH="$tool_dir/node_modules" node "$fixture_dir/minify.cjs"
