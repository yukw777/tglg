# real-time-vlm-benchmark

## Installation
```bash
# Install python dependencies
uv sync

# Workaround for https://github.com/OpenNMT/CTranslate2/issues/1826
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/
```
