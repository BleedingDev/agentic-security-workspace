#!/bin/bash
# Kaggle-matching environment on a Google Cloud TPU VM (v5litepod-8 or v6e-8). Idempotent: re-run after a preemption.
# Datasets go to a tmpfs in RAM by default (TMPFS_SIZE, default 160g; both hosts have far more RAM) or to a persistent
# disk you attached (DATA_DISK=/dev/sdb: formatted if empty, mounted at /mnt/data/kaggle/input).
set -euo pipefail
TMPFS_SIZE=${TMPFS_SIZE:-160g}; DATA_DISK=${DATA_DISK:-}
sudo mkdir -p /mnt/data && sudo chown "$USER:$USER" /mnt/data
mkdir -p /mnt/data/kaggle/input /mnt/data/kaggle/working /mnt/data/envs /mnt/data/logs
if ! mountpoint -q /mnt/data/kaggle/input; then
  if [ -n "$DATA_DISK" ]; then
    sudo blkid "$DATA_DISK" >/dev/null 2>&1 || sudo mkfs.ext4 -q "$DATA_DISK"
    sudo mount "$DATA_DISK" /mnt/data/kaggle/input && sudo chown "$USER:$USER" /mnt/data/kaggle/input
  else
    sudo mount -t tmpfs -o "size=$TMPFS_SIZE,mode=755,uid=$(id -u),gid=$(id -g)" tmpfs /mnt/data/kaggle/input
  fi
fi
mkdir -p /mnt/data/kaggle/input/datasets/rahim3
[ -e /kaggle ] || sudo ln -s /mnt/data/kaggle /kaggle      # the kernel globs /kaggle/input/datasets/<owner>/<slug>, writes /kaggle/working
command -v tmux >/dev/null && command -v unzip >/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq tmux unzip >/dev/null)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
[ -x /mnt/data/envs/glm/bin/python ] || uv venv --python 3.12 /mnt/data/envs/glm >/dev/null
source /mnt/data/envs/glm/bin/activate
uv pip install -q "jax[tpu]==0.10.2"                       # Kaggle's JAX
uv pip install -q "libtpu==0.0.42.*"                       # the kernel's libtpu pin (newer runtimes cannot run its Pallas kernels)
uv pip install -q safetensors huggingface_hub "transformers>=5.16" pillow ml_dtypes kaggle flatbuffers
uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
grep -q 'envs/glm/bin/activate' ~/.bashrc || printf '\nexport PATH="$HOME/.local/bin:$PATH"\nsource /mnt/data/envs/glm/bin/activate\n' >> ~/.bashrc
python - <<'PY'
import jax
d = jax.devices(); ms = d[0].memory_stats()
print(f"{len(d)} devices, {d[0].device_kind}, {ms['bytes_limit'] / 1e9:.2f} GB HBM per chip, jax {jax.__version__}")
assert len(d) == 8, "the recipe needs eight chips (v5litepod-8 or v6e-8)"
PY
echo "datasets space: $(df -h /mnt/data/kaggle/input | tail -1)"
