# ReF-WRD: Reliability-Aware Frequency Conditioning and Wavelet-Guided Residual Diffusion

This repository provides the core implementation and main evaluation protocol for **ReF-WRD**, the method described in the manuscript:

> Reliability-Aware Frequency Conditioning and Wavelet-Guided Residual Diffusion for Real-World Image Super-Resolution

The code base retains several internal file names using `fde_sr` from an earlier development stage. These files implement the ReF-WRD model components described in the manuscript. The naming mapping is:

```text
fde_sr.py                         ReF-WRD generator and inference wrapper
fde_sr_core.py                    OSEDiff-style one-step latent restoration backbone
models/fde_sr_modules.py          FDPL, residual correction, A-WRM, and losses
train_fde_sr_base.py              Stage-I base restoration training
train_fde_sr.py                   full / Stage-II residual-correction training entry
test_fde_sr.py                    ReF-WRD inference entry
test_metrics.py                   metric evaluation entry
```

## Main components

- **FDPL**: frequency-decoupled prompt learning with reliability gating.
- **Stage-I base restoration**: OSEDiff-style one-step latent restoration from the low-quality latent.
- **Stage-II residual correction / RC**: scale-controlled residual correction around the frozen Stage-I base. This branch does not introduce an additional diffusion reverse sampling trajectory.
- **A-WRM**: adaptive wavelet residual modulation in latent space.
- **WCS**: image-domain wavelet consistency supervision.

## Repository layout

```text
README.md                         main project overview
REPRODUCIBILITY.md                training, inference, metrics, and seed protocol
DATASETS.md                       dataset layout and list-file preparation
MODEL_ZOO.md                      external assets and checkpoint information
requirements.txt                  Python package requirements
configs/                          example protocol/configuration files
scripts/reproducibility/          documented reproduction scripts
checkpoints/README.md             checkpoint placement notes
```

Core code:

```text
fde_sr.py
fde_sr_core.py
models/fde_sr_modules.py
train_fde_sr_base.py
train_fde_sr.py
test_fde_sr.py
test_metrics.py
```

## Installation

```bash
conda create -n refwrd python=3.10 -y
conda activate refwrd
pip install --upgrade pip
pip install -r requirements.txt
```

Required external assets:

- Stable Diffusion 2.1 base directory for VAE, tokenizer, text encoder, scheduler, and UNet.
- RAM checkpoint.
- DAPE/RAM fine-tuned checkpoint for degradation-aware prompts.
- ReF-WRD checkpoint for inference, or a Stage-I base checkpoint for Stage-II training.

See `MODEL_ZOO.md` for checkpoint and asset notes.

## Dataset preparation

Benchmark datasets are not redistributed. Obtain LSDIR, FFHQ, DIV2K-Val, RealSR, and DRealSR from their original providers.

Generate training list files:

```bash
bash scripts/reproducibility/prepare_data.sh \
  /path/to/LSDIR/HR_merge \
  /path/to/FFHQ/images \
  data/lists
```

See `DATASETS.md` for the expected directory layout.

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

The main evaluation uses the fixed residual strength documented in the manuscript. In the public scripts this value is passed through `--fde_residual_scale`.

## Training

Stage-I base restoration:

```bash
export SD21BASE_PATH=/path/to/stable-diffusion-2-1-base
export RAM_PATH=/path/to/ram_swin_large_14m.pth
export TRAIN_TXT=data/lists/lsdir_train.txt
export FFHQ_TXT=data/lists/ffhq_first10k.txt
export SEED=42
bash scripts/reproducibility/run_stage1_base.sh
```

Stage-II residual correction:

```bash
export REFWRD_BASE_CKPT=/path/to/refwrd_stage1_base.pkl
export SEED=42
bash scripts/reproducibility/run_stage2_residual.sh
```

Repeat the documented training protocol with seeds `42`, `123`, and `2026` for repeated-run reporting.

## Metrics

```bash
bash scripts/reproducibility/run_metrics.sh \
  /path/to/restored_images \
  /path/to/gt_images \
  /path/to/metrics_log_dir
```

## Reproducibility notes

- The direct OSEDiff-style one-step baseline and ReF-WRD are evaluated under the same local protocol described in the manuscript.
- The cross-dataset residual-strength sweep is a post-hoc sensitivity analysis after fixing the main operating point and is not used for model selection or test-set parameter tuning.
- Stage-II residual correction does not run an additional diffusion reverse sampling trajectory.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
