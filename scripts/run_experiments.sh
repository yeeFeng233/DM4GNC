# script demo

# train
# python scripts/train.py --config_path configs/dm4gnc/cora.yml --stage_start vae_train --stage_end vae_encode --vae_name vae_class_v2
# visualization
# python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize vae_encode --vae_name vae_class_v2

# train diffusion
# python scripts/train.py --config_path configs/dm4gnc/cora.yml \
# --stage_start diff_sample \
# --stage_end vae_decode \
# --vae_name vae_class --vae_filter false \
# --diff_generate_ratio -1
# visualization
# python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --vae_name normal_vae
# python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --vae_name vae_class
# python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize diff_sample --diff_generate_ratio -1 --vae_name vae_class_v2

# filter samples
# python scripts/train.py --config_path configs/dm4gnc/cora.yml \
# --stage_start diff_sample \
# --stage_end diff_sample \
# --vae_name vae_class --diff_filter true \
# --diff_generate_ratio -1 \
# --filter_strategy "distance"
# visualization
# python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize filter_samples \
#  --diff_generate_ratio -1 --vae_name vae_class --filter_strategy "distance"


# filter_sample to classifier_test
# python scripts/train.py --config_path configs/dm4gnc/cora.yml \
# --stage_start classifier_train \
# --stage_end classifier_test \
# --vae_name vae_class --diff_filter true \
# --diff_generate_ratio 0 \
# --vae_threshold 0.97 \
# --filter_strategy "distance"
# visualization neighbor distribution from stage filter_samples
# python scripts/visualize_results.py --config_path configs/dm4gnc/cora.yml --stage_to_visualize neighbor_distribution \
# --diff_generate_ratio -1 --vae_name vae_class --filter_strategy "distance" --diff_filter true

# train vae in Computers
python scripts/train.py --config_path configs/dm4gnc/computers.yml \
--stage_start vae_train \
--stage_end vae_encode \
--vae_name vae_dec
# visualization
# python scripts/visualize_results.py --config_path configs/dm4gnc/computers.yml --stage_to_visualize vae_encode --vae_name normal_vae

