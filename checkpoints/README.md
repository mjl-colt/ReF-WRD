# Checkpoints

Large model files should not be committed directly to git. Put released checkpoints on the GitHub Release page or another stable file host, and document the download location in `MODEL_ZOO.md`.

Expected local layout for examples:

```text
preset/models/
  refwrd_stage1_base.pkl
  refwrd_stage2.pkl
```

The existing training and testing scripts accept checkpoint paths through arguments such as `--resume_fde_sr_path` and `--fde_sr_path`. These argument names are retained from the development code and correspond to ReF-WRD checkpoints.
