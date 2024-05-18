from einops import rearrange
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.data.utils.data_utils import NormalizationType
from dlimp import DLataset
import torch
import numpy as np
import datasets
from absl import logging
from torch.utils.data import IterableDataset
from torchvision import transforms
from torch import distributed as dist
from typing import Union, Sequence, Tuple, Dict, Any, List

# from memory_profiler import profile

class OXETorchDataset(IterableDataset):
  """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""
  def __init__(
      self,
      rlds_dataset: DLataset,
      dataset_statistics: dict = None,
      data_augmentation=True,
      random_erasing=False,
  ):
    self.rlds_dataset = rlds_dataset
    self.transform = lambda x: x
    self.iterations = 0
    if data_augmentation and random_erasing:
        self.transform = transforms.Compose([
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomApply([
                transforms.RandomErasing(p=0.5,
                                        scale=(0.02, 0.2),
                                        ratio=(0.3, 2.0),
                                        value=0,
                                        inplace=False),
                # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            ])
        ])
#   @profile
  def __iter__(self):
    shift = 0
    mod = 1
    if dist.is_initialized():
        shift = torch.cuda.current_device() if torch.cuda.is_available() else 0
        mod = torch.cuda.device_count() if torch.cuda.is_available() else 1
    worker_info = torch.utils.data.get_worker_info()
    if worker_info:
      mod *= worker_info.num_workers
      shift = self.rank * worker_info.num_workers + worker_info.id
    
    for i, sample in enumerate(self.rlds_dataset.as_numpy_iterator()):
      if (i + shift) % mod == 0:

        observation = rearrange(self.transform(torch.as_tensor(sample['observation']['image_primary'] / 255.0 , dtype=torch.float32)),
                                 'b h w c -> b c h w',
        )
        yield {
            'observation': {
                "image_primary": observation
            },
            'action': sample['action'],
            'language_instruction': sample['task']['language_instruction'].decode() ,
        }

def get_interleaved_oxe_dataset(name:  Union[str, Sequence[Tuple[str, float]]] = "ee_pose_magic_soup",
                                data_dir: str = "gs://gresearch/robotics",
                                train: bool = True,
                                data_augmentation=True,
                                future_action_window_size=0,
                                shuffle_buffer_size=500000,
                                num_parallel_calls=4,
                                num_threads=4) -> DLataset:
    
  dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
      name,
      data_dir,
      load_camera_views=("primary", "wrist"),
      action_proprio_normalization_type=NormalizationType.NONE,
  )
  logging.info("Creating interleaved OXE dataset {} from {}".format(name, data_dir))
  return make_interleaved_dataset(
      dataset_kwargs_list,
      sample_weights,
      train=train,
      shuffle_buffer_size=
      shuffle_buffer_size,  # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
      batch_size=None,  # batching will be handles in PyTorch Dataloader object
      balance_weights=True,
      traj_transform_kwargs=dict(
          goal_relabeling_strategy=None,
          window_size=6,
          future_action_window_size=future_action_window_size,
          subsample_length=50,
      ),
      frame_transform_kwargs=dict(
          image_augment_kwargs={
              "primary":
                  dict(
                      random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                      random_brightness=[0.1],
                      random_contrast=[0.9, 1.1],
                      random_saturation=[0.9, 1.1],
                      random_hue=[0.05],
                      augment_order=[
                          "random_resized_crop",
                          "random_brightness",
                          "random_contrast",
                          "random_saturation",
                          "random_hue",
                      ],
                  ),
          } if data_augmentation else {},
          resize_size=dict(primary=(224, 224),),
          num_parallel_calls=num_parallel_calls,
      ),
      traj_transform_threads=num_parallel_calls,
      traj_read_threads=num_threads,
  )


def get_hf_dataset(dataset_path: str = "jxu124/OpenX-Embodiment",
                   dataset_name: str = "fractal20220817_data",
                   split: str = "train",
                   streaming: bool = True):
  logging.info("Fetching dataset {}/{}".format(dataset_path, dataset_name))
  ds = datasets.load_dataset(dataset_path,
                             dataset_name,
                             streaming=streaming,
                             split=split,
                             cache_dir="dataset_cache")
  return ds
