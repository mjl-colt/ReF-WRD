# Dataset preparation

ReF-WRD uses public datasets that must be downloaded from their original
providers. This repository does not redistribute dataset images.

## Training sources

Prepare HR image folders:

```text
/path/to/LSDIR/HR_merge/
/path/to/FFHQ/images/
```

Generate list files:

```bash
bash scripts/reproducibility/prepare_data.sh \
  /path/to/LSDIR/HR_merge \
  /path/to/FFHQ/images \
  data/lists
```

The generated text files contain absolute paths:

```text
data/lists/lsdir_train.txt
data/lists/ffhq_first10k.txt
```

The Real-ESRGAN second-order degradation configuration is provided in:

```text
dataloaders/params_realesrgan.yml
```

## Evaluation benchmarks

Arrange paired benchmarks as:

```text
data/benchmarks/
  DIV2K-Val/
    LQ/
    HR/
  DRealSR/
    LQ/
    HR/
  RealSR/
    LQ/
    HR/
```

The exact folder names can be changed, but the paths passed to
`run_test_refwrd.sh` and `run_metrics.sh` should point to the corresponding LQ
and HR folders.

## Held-out validation set

The validation set used for checkpoint inspection and residual-strength
selection is constructed from training sources after synthetic degradation. It
must not overlap with DIV2K-Val, RealSR, or DRealSR test benchmarks.

