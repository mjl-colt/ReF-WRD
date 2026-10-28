# ReF-WRD implementation notes

`FDE_SR` is the internal development name retained by several code files. It
corresponds to the ReF-WRD method described in the manuscript. The active
implementation follows the manuscript while preserving old command names.

## Paper-to-code mapping

- **FDPL** (`models/ref_wrd_modules.py`): fixed Gaussian low/high-frequency
  decomposition, independent shallow CNN encoders, independent low-rank token
  adapters, and a token-wise reliability gate. The fused condition is normalized
  as `LN(t_low + lambda_g * gate * t_high)`.
- **Stage I** (`fde_sr.py`): one-step latent restoration conditioned by FDPL.
- **Stage II** (`LatentResidualCorrector`): predicts one residual in VAE latent
  space around a detached Stage-I base latent and detached FDPL condition.
- **A-WRM** (`AdaptiveWaveletResidualModulator`): non-decimated stationary Haar
  transform, a depth-wise LL residual processor, two channel-attention HF
  residual blocks, and four channel-wise subband gates.
- **WCS** (`wavelet_consistency_loss`): matches restored HF bands to the target
  and anchors the restored LL band to the detached Stage-I base image.

## Two-stage training

Stage I trains the one-step backbone and FDPL; the latent corrector is frozen:

```bash
accelerate launch train_fde_sr.py --training_stage stage1 [other arguments]
```

Stage II freezes the entire Stage-I path and FDPL, and trains only the latent
corrector with image L1, LPIPS, WCS, and latent-residual L1 objectives:

```bash
accelerate launch train_fde_sr.py \
  --training_stage stage2 \
  --resume_fde_sr_path=STAGE1_CHECKPOINT \
  --lambda_stage2_rec=1.0 \
  --lambda_stage2_lpips=0.2 \
  --lambda_fde_wavelet=1.0 \
  --lambda_fde_residual=0.5 \
  --wavelet_ll_weight=0.2 \
  --wavelet_high_weight=1.0 \
  [other arguments]
```

`--train_stage2_only` remains as an alias for `--training_stage stage2`.

## Inference

```bash
python test_fde_sr.py \
  -i INPUT_DIRECTORY \
  -o OUTPUT_DIRECTORY \
  --fde_sr_path CHECKPOINT \
  --pretrained_model_name_or_path SD21BASE_PATH \
  --ram_ft_path DAPE_PATH \
  --ram_path RAM_PATH \
  --save_intermediate
```

The optional intermediate outputs are the Stage-I base image and the decoded
effect of the latent residual. `--disable_stage2` provides the Stage-I ablation.

## Checkpoint layout

```python
{
    "format": "ref_wrd_v2",
    "base": {...},
    "ref_wrd_prompt": {...},
    "ref_wrd_stage2": {...},
    "fde_config": {...},
}
```

The loader also accepts the old `fde_prompt` and `fde_stage2` keys. Tensors that
do not match the paper-aligned architecture are skipped during non-strict load.

## Lightweight checks

```bash
python -m pytest test_fde_sr_modules.py -q
python -m py_compile models/ref_wrd_modules.py fde_sr.py train_fde_sr.py
```
