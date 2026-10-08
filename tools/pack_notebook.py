#!/usr/bin/env python3
"""Generate a model's run-all Kaggle notebook from its kernel script (and engine package, if it has one), so the
notebook and the kernel always ship the same code.

    python tools/pack_notebook.py glm53-flash      # writes glm53-flash/notebook/glm53-tpu-serve.ipynb

The notebook: an intro, a config cell (serve_config.json), one cell that writes the engine files as plain text
(nothing to edit there), the serving script, and the launch cell that is the server.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INTRO = {"glm53-flash": """# GLM-5.3-Flash on a free Kaggle TPU v5e-8

This notebook serves **GLM-5.3-Flash**, a 320B-parameter mixture-of-experts model (18B active), on Kaggle's free
TPU and gives you a public endpoint that speaks both the OpenAI and the Anthropic API. Use it from your laptop,
Claude Code, Codex CLI, opencode, or plain `curl`.

As far as we know this is the first inference engine to run GLM-5.3-Flash on a TPU, and it runs on the free one.
We wrote it in JAX for this model: the experts sit in HBM as 3-bit weights that custom TPU kernels read straight
into the matrix multiply, and a small draft model proposes blocks of tokens that the big model verifies in one step.
Measured here: about **165 tok/s** for one stream, about **170 tok/s** aggregate over three, the model's full
**262,144-token context**, prefix caching that resumes an agent session in about a second, and image inputs.

## Before you run: three clicks in the right sidebar
1. **Accelerator → TPU VM v5e-8**
2. **Internet → ON** (the tunnel needs it)
3. **Add Input** → attach these three datasets:
   - `rahim3/glm53-flash-iq3xxs-1` and `rahim3/glm53-flash-iq3xxs-2`: the experts (Unsloth's UD-IQ3_XXS GGUF)
   - `rahim3/glm53-flash-serve-v3`: the rest of the weights, the vision tower, the tokenizer and the programs

Then run all cells. The last cell is the server and keeps running on purpose; the endpoint URL and the API key
appear in its `READY` banner after about 6–13 minutes (the weights are most of that). The draft model
(`incoai/GLM-5.3-Flash-DFlash2`, CC BY-NC-ND 4.0, research use) comes from Hugging Face when the server starts;
set `dflash` to `""` in the config cell to run without it, at about 79 tok/s.

If the last cell stops in its first minute, its message says why: no TPU in this session (Kaggle does that
sometimes; stop and start the session again), Internet off, or a dataset not attached.

The code, the engine and a guide for the same recipe on a rented TPU VM are in the GitHub repo:
https://github.com/ARahim3/kaggle-tpu-lab. Keep an eye on it for more models in the future!
""",
}

CONFIG = {"glm53-flash": {"streams": 3, "max_len": 262144, "reasoning_effort_default": "low", "vision": True, "keepalive_min": 90}}

CONFIG_NOTES = {"glm53-flash": """### Configuration
The defaults above are what we serve. Things you might change:
- `streams`: how many requests decode together. Three fit in HBM at the full 262k context; a fourth request waits
  for a free slot. One stream runs at about 165 tok/s, three together at about 170 tok/s aggregate.
- `dflash`: where the draft model comes from: `hf:<repo>@<revision>` (the default, fetched from Hugging Face), an
  attached dataset or directory holding its `config.json` and `model.safetensors`, or `""` for no speculative
  decoding.
- `reasoning_effort_default`: `"low"`, `"medium"` or `"high"`; clients can still set it per request.
- `vision`: `false` skips the vision tower (a minute and a little HBM saved; image inputs then error out).
- `keepalive_min`: the server shuts itself down after this long. Raise it (up to about 480) for a long-lived
  endpoint; the default keeps a test run cheap on your TPU quota.
- `api_key`: set your own; otherwise one is generated and printed in the banner.
- `ntfy_topic`: optional; the script posts its progress to `ntfy.sh/<topic>` so you can follow it from your phone.
""",
}

LAUNCH_NOTES = {"glm53-flash": """### Launch (leave this cell running: it is the server)
What you will see:
1. **Runtime**: a pinned TPU runtime and a few packages (about a minute)
2. **Weights**: the 3-bit experts and the int8 non-expert weights go straight onto the eight chips (3–10 minutes,
   depending on how fast the dataset mount reads)
3. **Vision**: the vision tower (seconds), then the draft model from Hugging Face (about a minute)
4. **Warm-up**: every program is loaded in its exported and compiled form from the serve dataset (under a minute);
   without the matching dataset every program is traced and compiled (about 28 minutes)
5. **Tunnel**: a public URL
6. **READY banner** with `ENDPOINT`, `API KEY` and `MODEL`, then a short self-test

Use the endpoint from anywhere:
```bash
curl <ENDPOINT>/v1/chat/completions -H "Authorization: Bearer <API_KEY>" \\
  -H "Content-Type: application/json" -d '{
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
Claude Code (the banner prints this line filled in):
```bash
ANTHROPIC_BASE_URL=<ENDPOINT> ANTHROPIC_AUTH_TOKEN=<API_KEY> ANTHROPIC_MODEL=glm-5.3-flash \\
ANTHROPIC_SMALL_FAST_MODEL=glm-5.3-flash CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144 claude
```
""",
}


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def engine_cell(engine_dir: Path, pkg: str):
    files = sorted(p for p in engine_dir.glob("*.py"))
    parts = [f"# The engine: the `{pkg}` package ({len(files)} files), generated from the repo's {engine_dir.relative_to(ROOT)}/ — nothing to edit here.\n",
             "import os\n", f'os.makedirs("{pkg}", exist_ok=True)\n', "FILES = {}\n"]
    for p in files:
        src = p.read_text()
        assert "'''" not in src, f"{p} contains ''' — the engine cell cannot embed it"
        assert not src.endswith("\\"), p
        parts.append(f'FILES["{pkg}/{p.name}"] = r\'\'\'{src}\'\'\'\n')
    parts.append("for path, src in FILES.items():\n    open(path, \"w\").write(src)\n")
    parts.append(f'print(f"wrote {{len(FILES)}} files of the {pkg} package")\n')
    return "".join(parts)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "glm53-flash"
    folder = ROOT / model
    kernel = next((folder / "kernel").glob("serve_*.py"))
    engine_dirs = list((folder / "engine").glob("*/__init__.py")) if (folder / "engine").is_dir() else []
    cells = [cell("markdown", INTRO[model]),
             cell("code", "%%writefile serve_config.json\n" + json.dumps(CONFIG[model], indent=2) + "\n"),
             cell("markdown", CONFIG_NOTES[model])]
    for init in engine_dirs:
        cells.append(cell("markdown", "### The engine\nOne cell that writes the engine package next to the script. It is generated from the repo, so there is nothing to edit here; scroll past it.\n"))
        cells.append(cell("code", engine_cell(init.parent, init.parent.name)))
    cells.append(cell("markdown", "### The serving script\nThe next cell writes the serving script (the same one `launch.py` pushes; the source and docs are in the [kaggle-tpu-lab repo](https://github.com/ARahim3/kaggle-tpu-lab)).\n"))
    cells.append(cell("code", f"%%writefile {kernel.name}\n" + kernel.read_text()))
    cells.append(cell("markdown", LAUNCH_NOTES[model]))
    cells.append(cell("code", f"!python {kernel.name}\n"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = folder / "notebook" / f"{model.split('-')[0]}-tpu-serve.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {len(cells)} cells, {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
