# OPAL-Flow

OPAL-Flow is an orientation-aware pipeline for single-panicle rice anthesis time identification. It performs panicle tracking, pose-normalized cropping, super-resolution, PanicleTimeMAE inference, and final start/peak time aggregation.

## Pipeline

**Field images** → **YOLO-OBB tracking** → **Pose-normalized panicle crops** → **Super-resolution** → **PanicleTimeMAE** → **Tstart / Tpeak**

## Main Files of OPAL-Flow

- `main.py`
- `track.py`
- `super_resolution.py`
- `infer.py`
- `aggregate.py`
- `trackbest.pt`
- `panicletimebest.pt`
- `README.md`

## Requirements

OPAL-Flow uses a number of third-party libraries that may need to be installed in your Python or conda environment.

Recommended environment:

```text
python==3.10
torch==2.4.1
torchvision==0.19.1
transformers==4.46.3
numpy==1.26.4
pillow==10.4.0
tqdm==4.66.5
matplotlib==3.8.4
pandas==2.2.2
scipy==1.13.1
opencv-python==4.10.0.84
scikit-image==0.24.0
```

Install dependencies:

```bash
pip install torch==2.4.1 torchvision==0.19.1 transformers==4.46.3 numpy==1.26.4 pillow==10.4.0 tqdm==4.66.5 matplotlib==3.8.4 pandas==2.2.2 scipy==1.13.1 opencv-python==4.10.0.84 scikit-image==0.24.0 huggingface_hub ultralytics
```

For CUDA-enabled GPUs, install the PyTorch version matching your CUDA environment:

```text
https://pytorch.org/get-started/locally/
```

## Dataset and Models

The demo dataset is downloaded automatically from Hugging Face:

```python
HF_DATASET_REPO_ID = "njauyang/rice"
HF_DATASET_SUBDIR = "rice"
```

Place the trained model files in the project directory:

```text
trackbest.pt
panicletimebest.pt
```

`trackbest.pt` is used for panicle detection and tracking.  
`panicletimebest.pt` is used for PanicleTimeMAE inference.

VideoMAEv2 is loaded automatically from Hugging Face:

```python
VMAE_MODEL_DIR = "OpenGVLab/VideoMAEv2-Base"
```

The default super-resolution backend is Upscayl. Install it from:

```text
https://github.com/upscayl/upscayl/releases
```

Then check the paths in `main.py`:

```python
SR_EXE = Path(r"C:\Program Files\Upscayl\resources\bin\upscayl-bin.exe")
SR_MODELS_DIR = Path(r"C:\Program Files\Upscayl\resources\models")
```
