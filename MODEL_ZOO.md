# Model zoo and external assets

Large checkpoints are not stored directly in git. ReF-WRD uses external model assets and project checkpoints that should be placed locally before training or inference.

## Required external assets

| Asset | Description | Source |
| --- | --- | --- |
| Stable Diffusion 2.1 base | VAE, tokenizer, text encoder, scheduler, and UNet | Obtain from the official Stable Diffusion 2.1 release |
| RAM checkpoint | Recognize Anything Model checkpoint | Obtain from the official RAM release |
| DAPE/RAM fine-tuned checkpoint | Degradation-aware prompt extractor used by the restoration pipeline | Obtain from the corresponding official or project release |

## ReF-WRD checkpoints

| Checkpoint | Recommended local file name | Purpose | Availability |
| --- | --- | --- | --- |
| Stage-I base | `refwrd_stage1_base.pkl` | OSEDiff-style one-step base restoration | Provided through the project release page or available from the authors upon request |
| Stage-II full model | `refwrd_stage2.pkl` | ReF-WRD with residual correction branch | Provided through the project release page or available from the authors upon request |

Recommended local layout:

```text
preset/models/
  refwrd_stage1_base.pkl
  refwrd_stage2.pkl
```

The training and inference scripts accept checkpoint paths through arguments such as `--resume_fde_sr_path` and `--fde_sr_path`. The internal argument names follow the development code and correspond to ReF-WRD checkpoints.

## Checkpoint format

Expected checkpoint dictionary format:

```python
{
    "format": "fde_sr_v1",
    "base": {...},
    "fde_prompt": {...},
    "fde_stage2": {...},
    "fde_config": {...},
}
```

The internal key names follow the development code. They correspond to the FDPL, Stage-I base restoration, and Stage-II residual-correction modules described in the manuscript.

## Checksums

For externally hosted checkpoint files, SHA256 checksums should be recorded either in this file or in the corresponding GitHub Release notes.
