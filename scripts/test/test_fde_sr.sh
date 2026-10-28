python test_fde_sr.py \
    -i preset/datasets/test_dataset/input \
    -o preset/datasets/test_dataset/output_fde_sr \
    --fde_sr_path experience/fde_sr/checkpoints/ref_wrd_1.pkl \
    --pretrained_model_name_or_path SD21BASE_PATH \
    --ram_ft_path DAPE_PATH \
    --ram_path RAM_PATH \
    --save_intermediate
