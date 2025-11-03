# script demo
# train
python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start vae_train --stage_end classifier_test
# visualization
python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize vae_encode 