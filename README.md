# KD_RL_HDC

## Quick start

```bash
# 1. Convert NuScenes mini to SemanticKITTI format
python main.py --convert

# 2. Train the CNN feature extractor (saves to logs/SENet*)
python main.py --train-cnn

# 3. Train the HDC classifier on frozen CNN features (saves to logs/hdc.pth)
python main.py --train-hdc

# Or run everything end-to-end
python main.py --all
```

Edit the path constants at the top of `main.py` to match your data layout:

```python
NUSCENES_DIR = "/data/nuscenes" # raw NuScenes mini download
CONVERTED_DIR = "/data/nuscenes_kitti" # output of --convert
LOG_DIR = "logs"
```