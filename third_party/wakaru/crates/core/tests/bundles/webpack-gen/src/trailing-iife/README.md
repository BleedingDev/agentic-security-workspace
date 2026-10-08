# Trailing async startup reproduction

Webpack 5.101.3 in production mode inlines `main()` into an async IIFE after
three entry-scope dependency declarations. `webpack5-trailing-iife-min.config.cjs`
and `generate.sh` reproduce the checked-in `dist/wp5-trailing-iife-min/bundle.js`.
The `failLoad` switch exercises rejection without editing generated output.

From the repository root, after building the CLI, verify both successful DOM
rendering and fire-and-forget rejection behavior:

```sh
WAKARU="$PWD/target/debug/wakaru" node crates/core/tests/bundles/webpack-gen/verify-trailing-iife.mjs
```

The rejected async task must leave module loading successful and surface a
separate unhandled rejection. A top-level-await rewrite fails this check.
