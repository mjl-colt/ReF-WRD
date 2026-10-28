# ReF-WRD: Reliability-Aware Frequency Conditioning and Wavelet-Guided Residual Denoising

This repository provides the core implementation and main evaluation protocol
for **ReF-WRD**, the method described in the manuscript:

> Reliability-Aware Frequency Conditioning and Wavelet-Guided Residual Denoising
> for Real-World Image Super-Resolution

The code base retains several internal file names using `fde_sr` from an earlier
development stage. In this repository, these files implement the ReF-WRD model
components:

```text
fde_sr.py                         ReF-WRD generator and inference wrapper
ref_wrd.py                        canonical paper-named import aliases
fde_sr_core.py                    OSEDiff-style one-step latent restoration backbone
models/ref_wrd_modules.py         FDPL, latent correction, A-WRM, and WCS
models/fde_sr_modules.py          compatibility exports for old imports
train_fde_sr.py                   Stage-I / Stage-II training entry
test_fde_sr.py                    ReF-WRD inference entry
test_metrics.py                   metric evaluation entry
```

## Main components

- **FDPL**: frequency-decoupled prompt learning with reliability gating.
- **Stage-I base restoration**: OSEDiff-style one-step latent restoration from
  the low-quality latent.
- **Stage-II residual correction / RC**: a scale-controlled latent residual
  around the frozen Stage-I base and condition. This branch does not introduce
  an additional diffusion reverse sampling trajectory.
- **A-WRM**: adaptive wavelet residual modulation in latent space.
- **WCS**: image-domain wavelet consistency supervision.

## Installation

```bash
conda create -n refwrd python=3.10 -y
conda activate refwrd
pip install --upgrade pip
pip install -r requirements.txt
```

Required external assets:

- Stable Diffusion 2.1 base directory.
- RAM checkpoint.
- DAPE/RAM fine-tuned checkpoint for degradation-aware prompts.
- ReF-WRD Stage-I and Stage-II checkpoints for inference.

The benchmark datasets are not redistributed. Please obtain LSDIR, FFHQ,
DIV2K-Val, RealSR, and DRealSR from their original providers.

## Quick inference

```bash
export SD21BASE_PATH=/path/to/stable-diffusion-2-1-base
export RAM_PATH=/path/to/ram_swin_large_14m.pth
export DAPE_PATH=/path/to/DAPE.pth
export REFWRD_CKPT=/path/to/refwrd_stage2.pkl

bash scripts/reproducibility/run_test_refwrd.sh \
  /path/to/LQ_images \
  /path/to/output_dir
```

## Training and evaluation

For the complete documented protocol, see:

- `REPRODUCIBILITY.md`
- `DATASETS.md`
- `MODEL_ZOO.md`
- `configs/`
- `scripts/reproducibility/`

## License

This repository is released under the MIT License. See `LICENSE`.

