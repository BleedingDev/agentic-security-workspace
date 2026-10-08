#!/bin/bash
# The three public datasets into Kaggle's layout: the two GGUF expert parts (file by file, six downloads in parallel)
# and the serve dataset (non-expert weights, vision tower, tokenizer, compiled programs). ~130 GB; 10-30 min on the
# VM's network. Re-runnable: finished files are skipped. Needs a Kaggle token in ~/.kaggle (access_token or kaggle.json).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"; source /mnt/data/envs/glm/bin/activate
OWNER=rahim3; IN=/mnt/data/kaggle/input/datasets/$OWNER
SERVE=${SERVE_DATASET:-glm53-flash-serve-v3}
dl_file() {  # slug file
  local d=$IN/$1; mkdir -p "$d"
  [ -s "$d/$2" ] && return 0
  kaggle datasets download "$OWNER/$1" -f "$2" -p "$d" -q && { [ -f "$d/$2.zip" ] && (cd "$d" && unzip -q -o "$2.zip" && rm "$2.zip"); }
}
export -f dl_file; export IN OWNER
for slug in glm53-flash-iq3xxs-1 glm53-flash-iq3xxs-2; do
  kaggle datasets files "$OWNER/$slug" --page-size 200 -v 2>/dev/null | tail -n +2 | cut -d, -f1 | grep -v '^$' \
   | xargs -P 6 -I{} bash -c "dl_file $slug '{}'"
done
dl_path() {  # slug path/inside/dataset (the directories are kept; the whole-dataset archive is not always available)
  local d=$IN/$1/$(dirname "$2"); mkdir -p "$d"; local f=$(basename "$2")
  [ -s "$d/$f" ] && return 0
  kaggle datasets download "$OWNER/$1" -f "$2" -p "$d" -q && { [ -f "$d/$f.zip" ] && (cd "$d" && unzip -q -o "$f.zip" && rm "$f.zip"); }
}
export -f dl_path
list_files() {  # every file of a dataset: the listing comes in pages of at most 200
  local tok="" out
  while :; do
    out=$(kaggle datasets files "$1" --page-size 200 -v ${tok:+--page-token "$tok"} 2>/dev/null)
    echo "$out" | grep -v '^Next Page Token\|^name,' | cut -d, -f1 | grep -v '^$'
    tok=$(echo "$out" | sed -n 's/^Next Page Token = //p')
    [ -n "$tok" ] || break
  done
}
list_files "$OWNER/$SERVE" | xargs -P 4 -I{} bash -c "dl_path $SERVE '{}'"          # idempotent: existing files are skipped
echo "== done"; du -sh $IN/*
