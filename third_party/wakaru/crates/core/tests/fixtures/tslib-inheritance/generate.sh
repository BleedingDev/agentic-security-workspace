#!/usr/bin/env bash
set -euo pipefail
fixture_dir="$(cd "$(dirname "$0")" && pwd)"
tool_dir="$(mktemp -d)"
trap 'rm -rf "$tool_dir"' EXIT
npm install --prefix "$tool_dir" --no-audit --no-fund --no-package-lock \
  typescript@5.9.3 terser@5.51.2 >/dev/null
NODE_PATH="$tool_dir/node_modules" node "$fixture_dir/generate.cjs"
