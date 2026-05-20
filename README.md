---
title: Mci Model Mri
emoji: 🧬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

Model_MRI
=========

This repository contains the MRI model service for MCI detection.

Contents to copy into this repo before pushing:
- All files from `model_mri/` (e.g., `mri_api.py`, `train_kfold.py`, `preprocess.py`, pretrained model files in `pretrained/`).
- `aligned_outputs/` if using alignment smoke tests.

Quick start (local):

1. Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Flask API (assumes `mri_api.py` exists at repo root):

```powershell
python -m mri_api
# or
gunicorn -w 1 -b 0.0.0.0:5000 mri_api:app
```

Deploy notes:
- Set `MODEL_PRETRAINED_PATH` env var to point to the `.keras` or `.pth` model in `pretrained/`.
- Use GPU-enabled TensorFlow image for production if you want CUDA acceleration. See `Dockerfile`.

Repository push (example):

```bash
git init
git add .
git commit -m "Initial MRI model service"
git remote add origin git@github.com:AbhayRao38/Model_MRI.git
git push -u origin main
```

CI / Docker:
- A `Dockerfile` is provided to build a container. For GPU, base on `nvidia/cuda` and install `tensorflow==2.12.0` with GPU support.
