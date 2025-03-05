# real-time-vlm-benchmark

## Installation

```bash
# If Slurm, ensure you have the latest CUDA and GCC loaded
module load cuda gcc

# Create a virtual env, e.g., using uv
uv venv

# flash-attn build dependencies
uv pip install torch setuptools psutil packaging ninja

# Build flash-attn
uv pip install flash-attn --no-build-isolation

# Install python dependencies
# Specify --extra flash-attn if necessary
uv sync

# Workaround for https://github.com/OpenNMT/CTranslate2/issues/1826
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/
```
