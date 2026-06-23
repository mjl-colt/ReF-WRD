# ReF-WRD reproducibility guide

This document describes the public reproducibility protocol for ReF-WRD. The
benchmark datasets are not redistributed; users should obtain them from their
original providers and arrange them according to `DATASETS.md`.

## 1. Environment

```bash
conda create -n refwrd python=3.10 -y
conda activate refwrd
pip install --upgrade pip
pip install -r requirements.txt
```

The reported local training and runtime measurements use a single NVIDIA
GeForce RTX 4090 GPU unless otherwise specified in the manuscript.

## 2. External model assets

Set the following paths before training or testing:

```bash
export SD21BASE_PATH=/path/to/stable-diffusion-2-1-base
export RAM_PATH=/path/to/ram_swin_large_14m.pth
export DAPE_PATH=/path/to/dape_or_ram_ft_checkpoint.pth
```

For inference with released checkpoints:

```bash
export REFWRD_BASE_CKPT=/path/to/refwrd_stage1_base.pkl
export REFWRD_CKPT=/path/to/refwrd_stage2.pkl
```

## 3. Dataset list files

Training scripts expect text files containing absolute paths to HR images, one
image per line. Generate list files using:

```bash
bash scripts/reproducibility/prepare_data.sh \
  /path/to/LSDIR/HR_merge \
  /path/to/FFHQ/images \
  data/lists
```

This creates:

```text
data/lists/lsdir_train.txt
data/lists/ffhq_first10k.txt
```

## 4. Stage-I base restoration training

```bash
export TRAIN_TXT=data/lists/lsdir_train.txt
export FFHQ_TXT=data/lists/ffhq_first10k.txt
export SEED=42
bash scripts/reproducibility/run_stage1_base.sh
```

Repeat with seeds `42`, `123`, and `2026` for repeated-run reporting.

## 5. Stage-II residual correction training

```bash
export TRAIN_TXT=data/lists/lsdir_train.txt
export FFHQ_TXT=data/lists/ffhq_first10k.txt
export REFWRD_BASE_CKPT=experience/refwrd_stage1/checkpoints/refwrd_stage1_base.pkl
export SEED=42
bash scripts/reproducibility/run_stage2_residual.sh
```

Stage-II trains the residual correction branch around the Stage-I base. It does
not introduce an additional diffusion reverse sampling trajectory.

## 6. Inference

```bash
export REFWRD_CKPT=/path/to/refwrd_stage2.pkl
bash scripts/reproducibility/run_test_refwrd.sh \
  /path/to/benchmark/LQ \
  outputs/refwrd/benchmark_name
```

The main evaluation uses the fixed residual strength documented in the manuscript. In this code release, the corresponding inference argument is `--fde_residual_scale` and the reproducibility script sets `FDE_RESIDUAL_SCALE=0.25` by default. The cross-dataset residual-strength sweeps in the manuscript are post-hoc sensitivity analyses and are not used for model selection or test-set parameter tuning.

## 7. Metrics

```bash
bash scripts/reproducibility/run_metrics.sh \
  outputs/refwrd/benchmark_name \
  /path/to/benchmark/HR \
  outputs/refwrd/benchmark_name/metrics
```

## 8. Checkpoint selection

The main protocol uses a held-out validation set from the training sources after
synthetic degradation, with no overlap with DIV2K-Val, RealSR, or DRealSR test
benchmarks. Checkpoints are evaluated at the documented interval. Stage-I and
Stage-II checkpoints are selected according to the validation metrics described
in the manuscript. The test benchmarks are not used for checkpoint selection or
residual-strength selection.

## 9. Naming note

Several scripts retain the internal `fde_sr` name from development. These files
implement the ReF-WRD method described in the manuscript.



