# ReF-WRD: Reliability-Aware Frequency Conditioning and Wavelet-Guided Residual Denoising

This repository provides the core implementation and evaluation protocol for
**ReF-WRD**, a two-stage real-world image super-resolution method. Stage I uses
frequency-decoupled prompt learning (FDPL) for one-step latent restoration.
Stage II applies one adaptive wavelet-modulated correction around the frozen
Stage-I latent; it does not start another diffusion reverse trajectory.

The code retains several `fde_sr` filenames from an earlier development stage.
They now resolve to the paper-aligned ReF-WRD implementation.

## Main components

- **FDPL**: Gaussian low/high separation, independent CNN/token adapters, and
  token-wise reliability gating.
- **Stage-I base restoration**: OSEDiff-style one-step latent restoration from
  the low-quality latent.
- **Stage-II latent correction**: a single scale-controlled residual around
  detached Stage-I latent and condition tensors.
- **A-WRM**: non-decimated Haar SWT, dedicated LL/HF processing, and
  channel-wise LL/LH/HL/HH gates without spatial downsampling.
- **WCS**: high-frequency subbands are matched to the target, while the
  low-frequency subband is anchored to the detached Stage-I output.

Stage II is optimized with image L1, LPIPS, WCS, and an L1 latent-residual
magnitude penalty.

## Repository layout

```text
fde_sr.py                         generator and inference implementation
ref_wrd.py                        canonical paper-named import aliases
fde_sr_core.py                    one-step latent restoration backbone
models/ref_wrd_modules.py         FDPL, latent correction, A-WRM, and WCS
models/fde_sr_modules.py          compatibility exports for old imports
train_fde_sr.py                   paper-aligned Stage-I/Stage-II training entry
test_fde_sr.py                    ReF-WRD inference entry
test_fde_sr_modules.py            lightweight component tests
REPRODUCIBILITY.md                training, inference, metrics, and seed protocol
DATASETS.md                       dataset layout and list-file preparation
MODEL_ZOO.md                      external assets and checkpoint information
configs/                          example protocol/configuration files
scripts/reproducibility/          documented reproduction scripts
checkpoints/README.md             checkpoint placement notes
```

## Installation

```bash
conda create -n refwrd python=3.10 -y
conda activate refwrd
pip install --upgrade pip
pip install -r requirements.txt
```

Required external assets:

- Stable Diffusion 2.1 base directory for VAE, tokenizer, text encoder,
  scheduler, and UNet.
- RAM checkpoint.
- DAPE/RAM fine-tuned checkpoint for degradation-aware prompts.
- ReF-WRD checkpoint for inference, or a Stage-I checkpoint for Stage-II
  training.

See `MODEL_ZOO.md` for checkpoint and asset notes.

## Dataset preparation

Benchmark datasets are not redistributed. Obtain LSDIR, FFHQ, DIV2K-Val,
RealSR, and DRealSR from their original providers.

```bash
bash scripts/reproducibility/prepare_data.sh \
  /path/to/LSDIR/HR_merge \
  /path/to/FFHQ/images \
  data/lists
```

See `DATASETS.md` for the expected directory layout.

## Training

Stage I trains the one-step restoration adapters and FDPL while the latent
corrector remains frozen:

```bash
export SD21BASE_PATH=/path/to/stable-diffusion-2-1-base
export RAM_PATH=/path/to/ram_swin_large_14m.pth
export TRAIN_TXT=data/lists/lsdir_train.txt
export FFHQ_TXT=data/lists/ffhq_first10k.txt
export SEED=42
bash scripts/reproducibility/run_stage1_base.sh
```

Stage II freezes Stage I and FDPL and trains only the latent corrector:

```bash
export REFWRD_BASE_CKPT=/path/to/refwrd_stage1.pkl
export SEED=42
bash scripts/reproducibility/run_stage2_residual.sh
```

Repeat the documented protocol with seeds `42`, `123`, and `2026` for
repeated-run reporting.

## Inference

```bash
export SD21BASE_PATH=/path/to/stable-diffusion-2-1-base
export RAM_PATH=/path/to/ram_swin_large_14m.pth
export DAPE_PATH=/path/to/DAPE.pth
export REFWRD_CKPT=/path/to/refwrd_stage2.pkl

bash scripts/reproducibility/run_test_refwrd.sh \
  /path/to/LQ_images \
  /path/to/output_dir
```

`--save_intermediate` writes the Stage-I base image and a visualization of the
decoded latent correction. `--disable_stage2` provides the Stage-I ablation.

## Metrics

```bash
bash scripts/reproducibility/run_metrics.sh \
  /path/to/restored_images \
  /path/to/gt_images \
  /path/to/metrics_log_dir
```

## Reproducibility notes

- The direct OSEDiff-style one-step baseline and ReF-WRD use the same local
  evaluation protocol described in the manuscript.
- Residual strength is selected on held-out validation data. Cross-dataset
  sweeps are post-hoc sensitivity analyses, not test-set tuning.
- Stage-II correction does not run an additional diffusion reverse sampling
  trajectory.
- Old `fde_prompt`/`fde_stage2` checkpoint keys are accepted during non-strict
  loading; mismatched prototype tensors are skipped by shape.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
