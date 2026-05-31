# FDE_SR Implementation Notes

This note documents the FDE_SR code path in this repository. FDE_SR is organized
as a frequency-decoupled super-resolution model with a one-step latent
restoration backbone and a wavelet-modulated residual refinement stage.

## Main Components

- `models/fde_sr_modules.py`
  - `HaarWaveletTransform`
  - `FrequencyDescriptor`
  - `FrequencyDecoupledPromptLearner`
  - `StageIIResidualCorrector`
  - `AdaptiveWaveletResidualModulator`
  - `FDE_SRLoss`

- `fde_sr.py`
  - `FDE_SR_gen`
  - `FDE_SR_test`
  - `FDE_SR_inference_time`
  - checkpoint save/load helpers

- `train_fde_sr.py`
  - training loop with reconstruction, perceptual, distribution matching,
    residual alignment, high-frequency, and wavelet consistency losses

- `test_fde_sr.py`
  - inference entry with optional base/residual outputs

## Training

```bash
bash scripts/train/train_fde_sr.sh
```

For stage-II focused training:

```bash
bash scripts/train/train_fde_sr_stage2.sh
```

Important arguments:

```bash
--resume_fde_sr_path=preset/models/fde_sr_base.pkl
--lambda_fde_wavelet=0.5
--lambda_fde_residual=0.25
--lambda_fde_high=0.1
--fde_hidden_dim=128
--fde_prompt_hidden_dim=256
--fde_num_blocks=6
--fde_residual_scale=0.25
```

## Inference

```bash
python test_fde_sr.py \
    -i preset/datasets/test_dataset/input \
    -o preset/datasets/test_dataset/output_fde_sr \
    --fde_sr_path experience/fde_sr/checkpoints/fde_sr_1.pkl \
    --pretrained_model_name_or_path SD21BASE_PATH \
    --ram_ft_path DAPE_PATH \
    --ram_path RAM_PATH \
    --save_intermediate
```

Intermediate outputs:

- `output_fde_sr/base/`: one-step base restoration
- `output_fde_sr/residual/`: normalized residual visualization
- `output_fde_sr/`: final FDE_SR restoration

## Checkpoint Layout

```python
{
    "format": "fde_sr_v1",
    "base": {...},
    "fde_prompt": {...},
    "fde_stage2": {...},
    "fde_config": {...},
}
```

## Ablation Switches

```bash
--disable_stage2
--train_stage2_only
--lambda_fde_wavelet=0
--lambda_fde_residual=0
--lambda_fde_high=0
```
