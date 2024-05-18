# oxe_torch
Training and dataloading utilities and examples for OpenX Embodiment Datasets


conda create -n p11 python=3.11

Example flags that may or may not work: --model=rt1 --batch_size=8 --num_epochs=10 --lr=5e-5 --steps_per_epoch=1000 --project_name=rt1_bridge_oxe_75_nocheck_large_weight --matmul_precision=medium --lr_scheduler=cos --future_action_window_size=5 --strategy=ddp --checkpoint_frequency=5000 --num_parallel_calls 16 --num_threads 16 --shuffle_buffer_size 1000 --gradient_clip_val 10000.0 --seed 11 --precision 16 --local_datasets ../concatenated.hdf5 --oxe_batch_percentage 0.75 --oxe_datasets bridge --log_images --log_image_frequency 17 --image_tokens_size=64 --layer_size=512 --norm_actions=gaussian


[Resources](https://github.com/sebbyjp/octo/blob/e9da7c436d3937301d263108a2aa6fb69a868eba/octo/data/oxe/oxe_dataset_mixes.py#L191)
[Datasets Info](https://dibyaghosh.com/rtx_viz/#roboturk)

Fork of octo being used: octo@git+https://github.com/sebbyjp/octo.git@peralta

Smallest dataset to use: ucsd_kitchen_dataset_converted_externally_to_rlds
Finetuning dataset: pick_coke_can_place_left_of_spoon.hdf5

Dataset most similar to the finetuning dataset: bridge
How to use: python run.py --helpfull

**Note** tokenization with normalization may be broken. Action normalization is also iffy 
