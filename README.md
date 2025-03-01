# real-time-vlm-benchmark

## Installation

```bash
# flash-attn has to be installed separately due to some build issues
pip install flash-attn==2.7.4.post1 --no-build-isolation
# Install python dependencies
uv sync

# Workaround for https://github.com/OpenNMT/CTranslate2/issues/1826
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib/
```
