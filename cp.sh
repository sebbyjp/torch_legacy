#!/bin/bash

for dataset in {"gs://gresearch/robotics/fanuc_manipulation_v2/",\
"gs://gresearch/robotics/fractal20220817_data/",\
"gs://gresearch/robotics/furniture_bench_dataset_converted_externally_to_rlds/",\
"gs://gresearch/robotics/iamlab_cmu_pickup_insert_converted_externally_to_rlds/",\
"gs://gresearch/robotics/imperial_wrist_dataset/",\
"gs://gresearch/robotics/imperialcollege_sawyer_wrist_cam/",\
"gs://gresearch/robotics/jaco_play/",\
"gs://gresearch/robotics/kaist_nonprehensile_converted_externally_to_rlds/",\
"gs://gresearch/robotics/kuka/",\
"gs://gresearch/robotics/language_table/",\
"gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim/",\
"gs://gresearch/robotics/language_table_blocktoblock_4block_sim/",\
"gs://gresearch/robotics/language_table_blocktoblock_oracle_sim/",\
"gs://gresearch/robotics/language_table_blocktoblock_sim/",\
"gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim/",\
"gs://gresearch/robotics/language_table_blocktorelative_oracle_sim/",\
"gs://gresearch/robotics/language_table_checkpoints/",\
"gs://gresearch/robotics/language_table_separate_oracle_sim/",\
"gs://gresearch/robotics/language_table_sim/",\
"gs://gresearch/robotics/maniskill_dataset_converted_externally_to_rlds/",\
"gs://gresearch/robotics/mt_opt_rlds/",\
"gs://gresearch/robotics/mt_opt_sd/",\
"gs://gresearch/robotics/mutex_dataset/",\
"gs://gresearch/robotics/nyu_door_opening_surprising_effectiveness/",\
"gs://gresearch/robotics/nyu_franka_play_dataset_converted_externally_to_rlds/",\
"gs://gresearch/robotics/nyu_rot_dataset_converted_externally_to_rlds/",\
"gs://gresearch/robotics/open_x_embodiment_and_rt_x_oss/",\
"gs://gresearch/robotics/qut_dexterous_manpulation/",\
"gs://gresearch/robotics/robo_net/",\
"gs://gresearch/robotics/robot_vqa/",\
"gs://gresearch/robotics/roboturk/",\
"gs://gresearch/robotics/stanford_hydra_dataset_converted_externally_to_rlds/",\
"gs://gresearch/robotics/stanford_kuka_multimodal_dataset_converted_externally_to_rlds/",\
"gs://gresearch/robotics/stanford_mask_vit_converted_externally_to_rlds/",\
"gs://gresearch/robotics/stanford_robocook_converted_externally_to_rlds/",\
"gs://gresearch/robotics/taco_play/",\
"gs://gresearch/robotics/tokyo_u_lsmo_converted_externally_to_rlds/",\
"gs://gresearch/robotics/toto/",\
"gs://gresearch/robotics/ucsd_kitchen_dataset_converted_externally_to_rlds/",\
"gs://gresearch/robotics/ucsd_pick_and_place_dataset_converted_externally_to_rlds/",\
"gs://gresearch/robotics/uiuc_d3field/",\
"gs://gresearch/robotics/usc_cloth_sim_converted_externally_to_rlds/",\
"gs://gresearch/robotics/utaustin_mutex/",\
"gs://gresearch/robotics/utokyo_pr2_opening_fridge_converted_externally_to_rlds/",\
"gs://gresearch/robotics/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds/",\
"gs://gresearch/robotics/utokyo_saytap_converted_externally_to_rlds/",\
"gs://gresearch/robotics/utokyo_xarm_bimanual_converted_externally_to_rlds/",\
"gs://gresearch/robotics/utokyo_xarm_pick_and_place_converted_externally_to_rlds/",\
"gs://gresearch/robotics/viola/"}
 do
  echo "Copying $dataset"
  gsutil cp -r "$dataset" gs://mbodied/datasets/oxe
done



