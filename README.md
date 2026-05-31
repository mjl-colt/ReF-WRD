# FDE_SR: Frequency-Decoupled Enhancement for Real-World Image Super-Resolution

FDE_SR is a real-world image super-resolution model built around frequency-aware
conditioning and residual enhancement. The model uses a one-step latent
restoration backbone to produce a clean base image, then refines structure and
texture with a frequency-decoupled prompt learner and a second-stage residual
correction module.

## Model Overview

FDE_SR contains four main parts:

- **Frequency-Decoupled Prompt Learning**: extracts low-frequency and
  high-frequency statistics from the degraded input and injects them into text
  conditioning tokens.
- **One-Step Latent Base Restoration**: generates an initial high-resolution
  restoration in a single denoising step.
- **Stage-II Residual Correction**: predicts a residual map using the degraded
  input, the base restoration, and their high-frequency responses.
- **Adaptive Wavelet Residual Modulation**: decomposes the residual into Haar
  wavelet bands and adaptively modulates low/high-frequency components before
  reconstruction.

The training objective combines reconstruction, perceptual, distribution
matching, residual alignment, high-frequency, and wavelet consistency losses.

## Repository Layout

```text
fde_sr.py                         FDE_SR generator and inference wrapper
fde_sr_core.py                    one-step latent restoration backbone
models/fde_sr_modules.py          frequency prompt, residual correction, losses
train_fde_sr.py                   full FDE_SR training entry
test_fde_sr.py                    FDE_SR inference entry
test_fde_sr_modules.py            lightweight module tests
scripts/train/train_fde_sr.sh     full training script
scripts/train/train_fde_sr_stage2.sh
scripts/test/test_fde_sr.sh       inference script
```

The data loaders, RAM/DAPE prompt extractor, VAE tiling utilities, and metric
scripts are kept in their original folders.

## Installation

```bash
conda create -n FDE_SR python=3.10 -y
conda activate FDE_SR
pip install --upgrade pip
pip install -r requirements.txt
```

Required model assets:

- SD2.1 base model directory for VAE, tokenizer, text encoder, scheduler, and UNet.
- RAM checkpoint.
- DAPE/RAM fine-tuned checkpoint for degradation-aware prompts.
- FDE_SR checkpoint for inference, or a base checkpoint for stage-II training.

## Quick Inference

```bash
python test_fde_sr.py \
    -i preset/datasets/test_dataset/input \
    -o preset/datasets/test_dataset/output_fde_sr \
    --fde_sr_path preset/models/fde_sr.pkl \
    --pretrained_model_name_or_path SD21BASE_PATH \
    --ram_ft_path DAPE_PATH \
    --ram_path RAM_PATH \
    --save_intermediate
```

`--save_intermediate` writes:

- final restored images to `output_fde_sr/`
- base restoration to `output_fde_sr/base/`
- normalized residual visualization to `output_fde_sr/residual/`

## Train FDE_SR

Prepare a txt file where each line is an absolute path to an HR training image:

```text
LSDIR/HR_merge/0000001.png
LSDIR/HR_merge/0000002.png
...
```

Full training:

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3," accelerate launch train_fde_sr.py \
    --pretrained_model_name_or_path=SD21BASE_PATH \
    --ram_path=RAM_PATH \
    --resume_fde_sr_path=preset/models/fde_sr_base.pkl \
    --learning_rate=5e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps 500 \
    --mixed_precision='fp16' \
    --report_to "tensorboard" \
    --seed 123 \
    --output_dir=experience/fde_sr \
    --dataset_txt_paths_list 'YOUR TXT FILE PATH','YOUR TXT FILE PATH' \
    --dataset_prob_paths_list 1,1 \
    --lambda_l2=1 \
    --lambda_lpips=2 \
    --lambda_vsd=1 \
    --lambda_vsd_lora=1 \
    --lambda_fde_wavelet=0.5 \
    --lambda_fde_residual=0.25 \
    --lambda_fde_high=0.1 \
    --fde_hidden_dim=128 \
    --fde_prompt_hidden_dim=256 \
    --fde_num_blocks=6 \
    --fde_residual_scale=0.25 \
    --deg_file_path="params_realesrgan.yml" \
    --tracker_project_name "train_fde_sr"
```

Stage-II focused training:

```bash
bash scripts/train/train_fde_sr_stage2.sh
```

## Checkpoint Format

FDE_SR checkpoints are dictionaries:

```python
{
    "format": "fde_sr_v1",
    "base": {...},
    "fde_prompt": {...},
    "fde_stage2": {...},
    "fde_config": {...},
}
```

## Useful Ablations

Prompt-only training:

```bash
accelerate launch train_fde_sr.py ... --disable_stage2
```

Stage-II only:

```bash
accelerate launch train_fde_sr.py ... --train_stage2_only
```

Disable FDE-specific losses:

```bash
--lambda_fde_wavelet=0 --lambda_fde_residual=0 --lambda_fde_high=0
```

## Evaluation

```bash
python test_metrics.py \
    --inp_imgs preset/datasets/benchmark_realsr/results_fde_sr \
    --gt_imgs preset/datasets/benchmark_realsr/test_HR \
    --log preset/datasets/benchmark_realsr/results_fde_sr/metrics
```

## License

See [LICENCE](LICENCE).
