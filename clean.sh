#!/usr/bin/env bash
# Wipe runtime artifacts so the repo is GitHub-ready / fresh-installable.
# Keeps source, install scripts, Docker, docs assets, and models.yaml.
# Recreated by: ./install.sh
set -euo pipefail

cd "$(dirname "$0")" || exit 1

YES=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) YES=1 ;;
    -h|--help)
      echo "Usage: $0 [--yes]"
      echo "  Removes env/, models weights, datasets, outputs, sd-scripts,"
      echo "  __pycache__, and HF_TOKEN. Restores empty .gitkeep scaffolds."
      exit 0
      ;;
    *)
      echo "error: unknown argument: $arg (try --yes)" >&2
      exit 1
      ;;
  esac
done

size_of() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null | cut -f1
  else
    echo "(absent)"
  fi
}

echo "FluxGym-R clean — reclaimable paths:"
echo "  env/          $(size_of env)"
echo "  models/       $(size_of models)"
echo "  datasets/     $(size_of datasets)"
echo "  outputs/      $(size_of outputs)"
echo "  sd-scripts/   $(size_of sd-scripts)"
echo "  __pycache__/  $(size_of __pycache__)"
if [[ -f HF_TOKEN ]]; then
  echo "  HF_TOKEN      present"
else
  echo "  HF_TOKEN      (absent)"
fi
echo

if [[ "$YES" -ne 1 ]]; then
  read -r -p "Type yes to permanently delete these: " reply
  if [[ "$reply" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

echo "Removing runtime artifacts..."
rm -rf env sd-scripts __pycache__
rm -f HF_TOKEN

# Drop any nested bytecode under the repo (e.g. leftover after partial cleans)
find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true

rm -rf models datasets outputs
mkdir -p models/clip models/unet models/vae outputs datasets
: > models/.gitkeep
: > models/clip/.gitkeep
: > models/unet/.gitkeep
: > models/vae/.gitkeep
: > outputs/.gitkeep
: > datasets/.gitkeep

echo
echo "Done. Core product remains; run ./install.sh before launching again."
echo "Note: caption/hub models may still live under ~/.cache/huggingface (not touched)."
