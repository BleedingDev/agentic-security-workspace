#!/usr/bin/env bash
# TypeScript importHelpers reproductions. Tests consume the checked-in JS.
set -euo pipefail
fixture_dir="$(cd "$(dirname "$0")" && pwd)"
tool_dir="$(mktemp -d)"
trap 'rm -rf "$tool_dir"' EXIT
npm install --prefix "$tool_dir" --no-audit --no-fund --no-package-lock \
  typescript@5.9.3 tslib@2.8.1 >/dev/null
cp "$fixture_dir/src/load.ts" "$tool_dir/load.ts"
for target in es5 es2015; do
  node "$tool_dir/node_modules/typescript/bin/tsc" "$tool_dir/load.ts" \
    --target "$target" --module commonjs --moduleResolution node \
    --lib es2015 --importHelpers --outDir "$fixture_dir/$target"
done
