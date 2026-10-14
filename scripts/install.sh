#!/usr/bin/env bash
# Create the "roomdelta" conda environment (Linux / WSL2, NVIDIA GPU, CUDA 12.8 wheels).
# Usage: bash scripts/install.sh [env-name]
set -eo pipefail   # no -u: conda activation scripts reference unset variables
ENV="${1:-roomdelta}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

eval "$(conda shell.bash hook)"
conda create -y -q -n "$ENV" python=3.11
conda activate "$ENV"

# PyTorch with CUDA 12.8 (needed for RTX 50xx; works on older GPUs too)
pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Depth Anything 3 and LightGlue from source at pinned commits. --no-deps: their pins
# (e.g. numpy<2, xformers) conflict with this stack and are not needed at runtime.
pip install -q --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
pip install -q --no-deps "lightglue @ git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8"

pip install -q -r "$HERE/requirements.txt"
pip install -q --no-deps -e "$HERE"

echo
echo "done. Next:"
echo "  conda activate $ENV && roomdelta doctor"
