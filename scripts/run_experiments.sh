# script demo

# train
python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start vae_encode --stage_end vae_encode
# visualization
python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize vae_encode 