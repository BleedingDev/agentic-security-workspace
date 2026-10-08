# Contributing

Thanks for wanting to help. A few notes so your time is well spent.

- **Open an issue before a pull request.** Say what you want to change and why. Most things
  here were measured on the TPU before they shipped, and a short conversation first saves
  both of us from a PR that cannot land.
- **Small fixes** are welcome as direct PRs: a typo, a broken link, a wrong number with the
  measurement behind it.
- **New recipes** follow the existing shape: one folder per model with a README, the kernel,
  the generated notebook, and numbers measured on Kaggle's TPU with a note on how. Please
  propose the model in an issue first so we agree on the approach.
- **What will not be merged:** changing a recipe's default model or weights to a variant
  (abliterated, quantized, a third-party mirror), ports to GPUs or other platforms, and large
  rewrites that did not start as an issue. Forks are welcome for all of those, and we are
  happy to link to them from the README.
- **Bug reports:** attach the kernel log (the notebook's output, or `/kaggle/working/vllm.log`
  for the Qwen recipe) and say whether the session had a real TPU: `import jax;
  print(jax.device_count())` should print 8. Most reports so far were sessions that Kaggle
  started without one.
