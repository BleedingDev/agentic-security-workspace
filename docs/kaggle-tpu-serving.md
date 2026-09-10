# Serve an abliterated Qwen3.8-27B on a free Kaggle TPU and drive this workspace with it

This guide stands up a local-cost-free model endpoint and points OpenCode 2 at it, so the
Agentic Security Workspace (Ghidra, Burp, and the rest of the tool lanes) runs on a model
**you** host: **Qwen3.8-27B abliterated** on Kaggle's free TPU v5e-8, full bf16, up to the
model's native 262k context, as an OpenAI- and Anthropic-compatible endpoint.

The serving recipe lives in [`serving/`](../serving) and is a fork of
[ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) (MIT). The wrapper
`serving/ktl` reduces the whole flow to a handful of commands. The abliterated weights are
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated).

> Use only on assets you own or are explicitly authorized to test. An uncensored model does
> not change the law or this workspace's evidence-first rules; targets are still untrusted
> evidence, never instructions.

## What you need first

- A **Kaggle account with TPU access**. This needs **two** verifications under Kaggle →
  Settings, and both matter: **phone verification** *and* **identity verification** (Persona).
  Without identity verification Kaggle silently gives the kernel a CPU box instead of a TPU,
  and the server dies with `Insufficient devices for 2D mesh: found 1, expected 8`. The
  upstream README mentions only phone verification; this is the first deviation to know about.
- The free tier includes ~20 TPU-hours/week.
- [`uv`](https://docs.astral.sh/uv/) for the local venv, and OpenCode 2 (`opencode2`).
- For the Ghidra MCP lane: a local Ghidra install (set `GHIDRA_DIR` if it is not under
  `/Applications/ghidra_*_PUBLIC`) with the GhidraMCP extension available.

## One-time setup

```bash
cd serving
./ktl setup                          # creates .venv and installs the Kaggle CLI
.venv/bin/python -m kaggle auth login --force
```

For `auth login`, open the printed `oauth2/authorize` URL in a browser and approve. Use the
**localhost-callback** flow (the default when a browser is available); the typed
verification-code prompt is broken in current Kaggle CLI versions, so let it catch the
callback rather than pasting a code.

## Start it, use it, stop it

```bash
./ktl start                          # abliterated model (default)
#   --stock            serve the stock Qwen3.8-27B (mirrored dataset, faster start)
#   --hf-model ORG/REPO  serve any other Qwen3.8-27B-architecture bf16 repo
#   --ghidra           also start the headless Ghidra MCP server (see below)
```

`start` pushes the kernel and waits through the two unavoidable Kaggle waits: the TPU
**queue** (minutes to ~90 for a free slot) and then ~20–25 minutes to load 55 GB of weights
and compile the TPU graphs. It prints the model id when the endpoint is live and writes the
URL and key where OpenCode 2 reads them.

**OpenCode 2 does not start the endpoint.** Run `./ktl start` first; only then:

```bash
cd ..                                # the workspace root, so the security skills + MCP load
opencode2 -m kaggle-tpu/qwen3.8-27b-abliterated
```

When you are done, free your weekly quota:

```bash
cd serving && ./ktl stop             # ends the Kaggle TPU session and the Ghidra MCP server
```

Other subcommands: `./ktl status`, `./ktl validate`, `./ktl measure`, `./ktl agentic`,
`./ktl ghidra {start|stop|status}`.

## Wire it into OpenCode 2

The endpoint's tunnel URL and API key are regenerated on every launch, so the provider reads
them from two files that `ktl` keeps current:

- `~/.config/kaggle-tpu-lab/base_url`
- `~/.config/kaggle-tpu-lab/api_key`

Add this provider once to your OpenCode 2 config (`~/.config/opencode2/opencode.json`). It
contains no secrets — only file references, resolved at load time:

```json
{
  "providers": {
    "kaggle-tpu": {
      "name": "Kaggle TPU (Qwen3.8-27B)",
      "package": "aisdk:@ai-sdk/openai-compatible",
      "settings": {
        "baseURL": "{file:~/.config/kaggle-tpu-lab/base_url}",
        "apiKey": "{file:~/.config/kaggle-tpu-lab/api_key}"
      },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B bf16 (Kaggle TPU)",
          "capabilities": { "tools": true, "reasoning": true, "input": ["text", "image"], "output": ["text"] },
          "compatibility": { "reasoningField": "reasoning_content" },
          "limit": { "context": 262144, "output": 65536 }
        },
        "qwen3.8-27b-abliterated": {
          "name": "Qwen3.8-27B abliterated bf16 (Kaggle TPU)",
          "capabilities": { "tools": true, "reasoning": true, "input": ["text", "image"], "output": ["text"] },
          "compatibility": { "reasoningField": "reasoning_content" },
          "limit": { "context": 262144, "output": 65536 }
        }
      }
    }
  }
}
```

The OpenCode 2 background service caches resolved `{file:}` values; `ktl` touches the config
after each launch so the daemon re-reads the new URL and key. MCP tools are exposed through
OpenCode 2's Code Mode (on by default), so a large tool server does not flood the context
with per-tool schemas.

The endpoint is standard OpenAI (`/v1`) and also exposes an Anthropic-compatible
`/v1/messages`, so Codex CLI, Claude Code, aider, etc. work too — see
[`serving/README.upstream.md`](../serving/README.upstream.md) for those and for the tuning
flags (`--max-model-len`, `--max-num-seqs`, `--reasoning-effort`, `--text-only`, `--fast-start`).

## The Ghidra MCP lane, headless (no GUI)

The `ghidra` tool lane needs a running Ghidra that speaks the GhidraMCP protocol. You do not
need the desktop app: GhidraMCP 6.x ships a headless server that loads a binary directly.

```bash
./ktl ghidra start /path/to/binary   # defaults to the workspace's sample target
./ktl ghidra status                  # up :8089
./ktl ghidra stop
```

`ktl` installs the GhidraMCP extension into `"$GHIDRA_DIR"/Ghidra/Extensions/` (a Ghidra
upgrade drops installed extensions; re-run `start` to reinstall) and launches
`GhidraMCPHeadlessServer --file <binary> --port 8089`. The workspace's `ghidra` MCP bridge
connects to `127.0.0.1:8089`. Then, from the workspace root, the model can drive Ghidra
through the connected MCP tools, or use `analyzeHeadless` and `objdump`/`nm`/`strings`
directly — both paths are validated by `./ktl agentic`.

## How this differs from upstream kaggle-tpu-lab

- **`serving/ktl`** — a single entrypoint (`setup`/`start`/`stop`/`status`/`sync`/`validate`/`measure`/`agentic`/`ghidra`) around the upstream `launch.py`.
- **`launch.py --hf-model ORG/REPO` / `--served-model-name`** — serve arbitrary Qwen3.8-27B-architecture bf16 weights (e.g. the abliterated model) by downloading them in-kernel instead of mounting the pinned mirror dataset. The abliterated repo keeps the MTP draft head, so speculative decoding stays on and the shipped XLA cache still matches (the cache is keyed on `[context, sequences, mtp, text_only]`, not the weights).
- **OpenCode 2 provider wiring** — `{file:}`-based URL/key with a config touch to force reload, and a `ktl sync` that reads the launcher's ntfy event stream to find the live tunnel URL.
- **Headless Ghidra MCP** — install the extension into `Ghidra/Extensions/` and run `GhidraMCPHeadlessServer --file`, so the workspace's `ghidra` lane works with no desktop session.
- **Kaggle prerequisites** — identity verification (not just phone) is required, or the TPU is silently downgraded to CPU.

## Credits

- Serving recipe: [ARahim3/kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab) (MIT), kept in [`serving/`](../serving) with its original license and README.
- Model weights: [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated); base model [Qwen3.8-27B](https://huggingface.co/Qwen) (Apache-2.0).
- Runtime: [vLLM](https://github.com/vllm-project/vllm) and [tpu-inference](https://github.com/vllm-project/tpu-inference); free TPUs from Kaggle.
