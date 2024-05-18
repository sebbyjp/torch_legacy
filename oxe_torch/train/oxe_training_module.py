from copy import copy
from lightning import seed_everything
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import wandb

from absl import app, flags
from collections import OrderedDict
from einops import rearrange, repeat
from gym import spaces

from lightning.pytorch import LightningModule, LightningDataModule, Trainer
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.loggers import WandbLogger

# from memory_profiler import profile

from torch import optim, nn
# from torch import distributed as dist
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from typing import Optional

from oxe_torch.action_tokenization import RTX1ActionTokenizer as ActionTokenizer
from oxe_torch.data.hd5_dataset import get_hd5_files_in, MbodiedDataset as HD5Dataset
from oxe_torch.data.oxe_dataset import OXETorchDataset, get_interleaved_oxe_dataset
from oxe_torch.mt1.rtx1 import RTX1, RT1Config, ViTConfig
from oxe_torch.rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from oxe_torch.rt1.transformer_network import TransformerNetwork
from oxe_torch.train.util import log_metrics

FLAGS = flags.FLAGS

flags.DEFINE_string('project_name', 'oxe', 'Wandb project')

# Data flags
flags.DEFINE_list("local_datasets", [], "Fine tune dataset file paths or directories.")
flags.DEFINE_bool("oxe", True, "Use OXE dataset.")

# utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds
flags.DEFINE_list("oxe_datasets", "fractal20220817_data", "Dataset name.")
flags.DEFINE_integer("shuffle_buffer_size", 1000, "Shuffle buffer size.")
flags.DEFINE_float('oxe_batch_percentage', 0.83, 'Percentage of batch to use for oxe')
flags.DEFINE_bool('shuffle', True, 'Shuffle dataset')
flags.DEFINE_integer("skip_iterations", 0, "Iterations to skip in the dataset.")
flags.DEFINE_integer("reload_every_n_steps", 0, "Reload every n steps.")

flags.DEFINE_integer("seed", None, "Random seed.")
flags.DEFINE_bool("data_augmentation", True, "Whether or not to use data augmentation.")
flags.DEFINE_bool("random_erasing", False, "Random image shape masking.")

# Training flags
flags.DEFINE_integer("overfit_batches", 0, "Overfit batches.")
flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
flags.DEFINE_integer("max_steps", int(1e6), "Max steps to train for.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_float('steps_per_epoch', 1000, 'Steps per epoch')
flags.DEFINE_string('lr_scheduler', 'cos', 'Scheduler [cos, one_cycle]')
flags.DEFINE_float("lr", 1e-4, "Learning Rate.")
flags.DEFINE_float("min_lr", 0, "Min Learning Rate.")
flags.DEFINE_float("weight_decay", 0, "Weight Decay.")
flags.DEFINE_float("gradient_clip_val", 1.0, "Gradient clip value.")
flags.DEFINE_integer("accumulate_grad_batches", 1, "Accumulate grad batches.")
flags.DEFINE_string("norm_actions", None, "Normalize actions [across_actions, gaussian, None]")
# Checkpoint flags
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint or 'last'.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory to save to.")
flags.DEFINE_string('resume', '', 'Resume [all, dataloader]')
flags.DEFINE_integer("checkpoint_frequency", 100, "Checkpoint frequency in steps.")
flags.DEFINE_integer("log_frequency", 50, "Log frequency in steps.")
flags.DEFINE_bool("log_images", False, "Log images.")
flags.DEFINE_integer("log_image_frequency", 5000, "Log image frequency in steps.")

# Model flags
flags.DEFINE_string('model', 'rt1', 'Model [rt1, rtx1]')
flags.DEFINE_integer('observation_history_size', 6, 'Observation history size')
flags.DEFINE_integer('future_action_window_size', 5, 'Future action window size')
flags.DEFINE_bool('causal_attention', True, 'Causal attention')
flags.DEFINE_integer('summary_depth', 1, 'Summary depth')
#RTX1 flags
flags.DEFINE_bool("freeze_vit", False, "Freeze ViT weights.")
flags.DEFINE_bool("pretrained_vit", True, "Use pretrained ViT weights.")
flags.DEFINE_bool("bias", False, "Use bias.")

#RT1 flags
flags.DEFINE_integer('token_embedding_dim', 512, 'Token embedding dimension')
flags.DEFINE_integer('num_layers', 8, 'Number of layers')
flags.DEFINE_integer('layer_size', 128, 'Layer size')
flags.DEFINE_integer('num_heads', 8, 'Number of heads')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')
flags.DEFINE_integer('image_tokens_size', 8, 'Image tokens size')

# Accelerator flags
flags.DEFINE_integer('num_parallel_calls', 4, 'Number of parallel calls')
flags.DEFINE_integer('num_threads', 4, 'Number of threads')
flags.DEFINE_string('matmul_precision', 'highest', 'Matmul precision [highest, high, medium]')
flags.DEFINE_bool('cpu', False, 'Use CPU')
flags.DEFINE_bool("profile", False, "Profile the training run.")
flags.DEFINE_string('sharding_strategy', 'FULL_SHARD', 'Sharding strategy')
flags.DEFINE_string('strategy', 'auto', 'Strategy [auto, fsdp, ddp, ddp2]')
flags.DEFINE_string('precision', None, 'Precision [16-mixed, 32]')

tf.config.set_visible_devices([], "GPU")

TEXT_ENCODER = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")


# set lightning device to cpu if no gpu is available
def dict_to_device(dict_obj, device, dtype=None):
  """
      put all the values in the [dict_obj] to [device]
     """
  for k, v in dict_obj.items():
    if isinstance(v, dict):
      dict_obj[k] = dict_to_device(v, device, dtype)
    elif not isinstance(v, torch.Tensor):
      dict_obj[k] = torch.tensor(v, device=device, dtype=dtype)
    else:
      dict_obj[k] = v.to(device, dtype=dtype)
  return dict_obj


class CombinedOXEHD5DataModule(LightningDataModule):

  def __init__(self,
               *,
               batch_size: int = 32,
               datapaths: list = [],
               shuffle: bool = True,
               seed: int = 42,
               embed_instructions=True,
               data_augmentation=True,
               random_erasing=False,
               future_action_window_size=5,
               oxe=True,
               oxe_batch_percentage=0.83,
               oxe_datasets='',
               shuffle_buffer_size=1000,
               num_parallel_calls=4,
               num_threads=4,
               reload_every_n_steps: int = 0,
               **kwargs):
    super().__init__(**kwargs)
    self.batch_size = batch_size
    self.datapaths = get_hd5_files_in(datapaths)
    self.shuffle = shuffle
    self.seed = seed
    self.num_parallel_calls = num_parallel_calls
    self.num_threads = num_threads
    self.embed_instructions = embed_instructions
    self.data_augmentation = data_augmentation
    self.oxe = oxe
    self.future_action_window_size = future_action_window_size
    self.oxe_batch_percentage = oxe_batch_percentage
    self.oxe_datasets = oxe_datasets
    self.shuffle_buffer_size = shuffle_buffer_size
    self.random_erasing = random_erasing
    self.reload_every_n_steps = reload_every_n_steps

    if not self.data_augmentation:
      self.transform = transforms.Compose([transforms.ToTensor()])
    else:
      self.transform = None

    if len(self.datapaths) == 0:
      self.oxe = True
      self.oxe_batch_size = self.batch_size
    elif self.oxe:
      self.oxe_batch_size = int(round(self.batch_size * self.oxe_batch_percentage))
      self.datasets_batch_size = int(
          round((self.batch_size - self.oxe_batch_size) / len(self.datapaths)))
    else:
      self.oxe_batch_size = 0
      self.datasets_batch_size = int(
          round((self.batch_size - self.oxe_batch_size) / len(self.datapaths)))

    self.configure_oxe()

  def configure_oxe(self):
    if self.oxe:
      oxe_dlimp_ds, statistics, _ = get_interleaved_oxe_dataset(
          self.oxe_datasets,
          train=True,
          data_augmentation=self.data_augmentation,
          shuffle_buffer_size=self.shuffle_buffer_size,
          future_action_window_size=self.future_action_window_size,
          num_parallel_calls=self.num_parallel_calls,
          num_threads=self.num_threads,
      )
      action_mean = sum([stat['action']['mean'] / len(stat) for stat in statistics])
      action_std = sum([stat['action']['std'] / len(stat) for stat in statistics])
      print(f'action mean= {action_mean}, action std= {action_std}')
      oxe_ds = OXETorchDataset(
          oxe_dlimp_ds,
          data_augmentation=self.data_augmentation,
          random_erasing=self.random_erasing,
      )
      self.oxe_dl = DataLoader(
          oxe_ds,
          batch_size=self.oxe_batch_size,
          num_workers=0,
          pin_memory=True,
      )

  def setup(self, stage=None):
    if hasattr(self, 'datasets'):
      del self.datasets
    self.datasets = []
    rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    for datapath in self.datapaths:
      dataset = HD5Dataset(
          datapath,
          shuffle=self.shuffle,
          data_augmentation=self.data_augmentation,
          random_erasing=self.random_erasing,
          future_action_window_size=self.future_action_window_size,
          rank=rank,
          world_size=world_size,
      )
      self.datasets.append((dataset, self.datasets_batch_size))

  def train_dataloader(self):
    if not hasattr(self, 'datasets'):
      self.setup()
    dataloaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
        ) for dataset, batch_size in self.datasets
    ]
    if self.oxe:
      dataloaders.append(self.oxe_dl)

    if len(dataloaders) == 1:
      return dataloaders[0]
    elif len(dataloaders) > 1:
      return CombinedLoader(dataloaders, mode='max_size_cycle')
    else:
      raise ValueError('No dataloaders found')


#@profile
def load_batch(batch,
               window_size,
               total_window_size,
               with_readout_tokens=False,
               embed_instructions=False):
  with torch.no_grad():
    language_instruction = None

    if isinstance(batch, list):
      if isinstance(batch[0]['language_instruction'], list):
        language_instruction = sum([batch[i]['language_instruction'] for i in range(len(batch))],
                                   [])
      elif isinstance(batch[0]['language_instruction'], torch.Tensor):
        language_instruction = torch.cat(
            [batch[i]['language_instruction'].detach() for i in range(len(batch))], dim=0)

      video = torch.concat([
          batch[i]['observation']['image_primary'][:, :window_size, :, :, :].detach()
          for i in range(len(batch))
      ],
                           dim=0)
      action = torch.concat([
          batch[i]['action'][:, window_size - 1:total_window_size, :].detach()
          for i in range(len(batch))
      ],
                            dim=0)

    else:
      language_instruction = batch['language_instruction']
      video = batch['observation']['image_primary'][:, :window_size, :, :, :]
      action = batch['action'][:, window_size - 1:total_window_size, :].detach()

    if embed_instructions:
      language_instruction = torch.as_tensor(TEXT_ENCODER(language_instruction).numpy())

    # TODO(speralta): Figure out how to add readout tokens
    # if with_readout_tokens:
    #   readout = torch.as_tensor(TEXT_ENCODER(['x', 'y', 'z', 'roll', 'pitch', 'yaw']).numpy())
    #   readout = repeat(readout, 'n e -> b f n e', b=video.shape[0], f=window_size)
    #   return video, language_instruction, action, readout

    return video, language_instruction, action


class RTX1Module(LightningModule):

  def __init__(self, config=None):
    super().__init__()

    self.config = default_config()
    if config is not None:
      self.config.update(config)
    self.action_tokenizer = ActionTokenizer()
    self.precision = self.config['precision']
    self.total_steps = self.config['num_epochs'] * self.config['steps_per_epoch']
    self.max_lr = self.config['lr']
    self.accuracy = Accuracy(task='multiclass', num_classes=256)
    self.log_images = self.config['log_images']
    self.criterion = nn.CrossEntropyLoss()
    self.window_size = 6
    self.future_action_window_size = self.config['future_action_window_size']
    self.total_window_size = self.window_size + self.future_action_window_size
    self.lr = self.config['lr']
    self.log_image_frequency = self.config['log_image_frequency']
    self.batch_size = self.config['batch_size']
    self.weight_decay = self.config['weight_decay']
    self.log_frequency = self.config['log_frequency']
    self.data_augmentation = self.config['data_augmentation']
    self.datasets = ', '.join(self.config['local_datasets'])
    self.norm_actions = self.config['norm_actions']
    if self.config['oxe']:
      self.datasets += ',' + ','.join(self.config['oxe_datasets'])
    self.save_hyperparameters({
        'lr': self.lr,
        'max_lr': self.max_lr,
        'total_steps': self.total_steps,
        'window_size': self.window_size,
        'future_action_window_size': self.future_action_window_size,
        'log_frequency': self.log_frequency,
        'datasets': self.datasets,
        'data_augmentation': self.data_augmentation,
        'batch_size': self.batch_size,
        'weight_decay': self.weight_decay,
        'resume': self.config['resume'],
        'shuffle': self.config['shuffle'],
        'shuffle_buffer_size': self.config['shuffle_buffer_size'],
        'random_erasing': self.config['random_erasing'],
        'norm_actions': self.norm_actions,
    })
    self.num_reloads = 0
    if self.config['cpu']:
      self.configure_model()

  def configure_model(self) -> None:
    if hasattr(self, 'model'):
      return
    vit_config = ViTConfig(pretrained=self.config['pretrained_vit'])
    action_mean = None
    action_std = None
    action_dim = 7
    if self.config['norm_actions'] == 'gaussian':
      action_std = self.action_tokenizer.action_std
      action_mean = self.action_tokenizer.action_mean
    elif self.config['norm_actions'] == 'across_actions':
      action_dim = 9

    rt1_config = RT1Config(num_actions=action_dim,
                           bias=self.config['bias'],
                           causal_attention=self.config['causal_attention'],
                           action_mean=action_mean,
                           action_std=action_std)
    self.model = RTX1(rt1_config=rt1_config, vit_config=vit_config)
    if self.config['freeze_vit']:
      for param in self.model.vit.parameters():
        param.requires_grad = False

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(),
                                 lr=self.config['lr'],
                                 weight_decay=self.config['weight_decay'])
    if self.config['lr_scheduler'] == 'cos':
      lr_config = {
          'scheduler':
              optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=int(self.total_steps),
                                                             T_mult=10,
                                                             eta_min=self.config['min_lr']),
          'interval':
              'step',
          'frequency':
              1
      }
    else:
      lr_config = {
          'scheduler':
              optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=self.config['lr'],
                                            total_steps=int(self.total_steps)),
          'interval':
              'step',
          'frequency':
              1
      }
    return [optimizer], [lr_config]

  def on_train_epoch_start(self) -> None:
    if self.config['skip_iterations'] > 0 and self.config['seed'] is not None:
      seed_everything(self.config['seed'] + self.num_reloads)
    for _ in range(self.config['skip_iterations']):
      next(iter(self.trainer.train_dataloader))

    print(f'Skipped {self.config["skip_iterations"]} iterations')

  #@profile
  def training_step(self, batch, batch_idx):
    self.accuracy = self.accuracy.to(self.device)
    self.model = self.model.to(self.device)
    video, instructions, action = load_batch(batch,
                                             self.window_size,
                                             self.total_window_size,
                                             embed_instructions=False)
    video = video.to(self.device)
    action = action.to(self.device)
    del batch

    outs = self.model.train_step(video, instructions)[:, :self.future_action_window_size + 1, :, :]
    out_preds = torch.argmax(outs, -1).to(self.device)

    ground_truth = self.action_tokenizer.tokenize_to_xyzrpyg(
        action, self.config['norm_actions']).to(self.device)

    loss = self.criterion(rearrange(outs, 'b f a bins -> (b f a) bins'),
                          rearrange(ground_truth, 'b f a -> (b f a)'))
    acc = self.accuracy(out_preds, ground_truth)
    log_metrics(
        self,
        loss.item(),
        acc,
        out_preds.cpu().detach(),
        ground_truth.cpu().detach(),
        video.cpu().detach() if self.log_images else None,
        instructions=instructions,
        batch_idx=batch_idx,
    )
    return loss


class RT1Module(RTX1Module):

  def __init__(self, config=None):
    super().__init__(config)

  def configure_model(self) -> None:
    if hasattr(self, 'model'):
      return
    observation_space = spaces.Dict({
        'image_primary':
            spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32),
        'natural_language_embedding':
            spaces.Box(low=-np.inf, high=np.inf, shape=[512], dtype=np.float32)
    })
    action_space_dict = OrderedDict([
        # ("done", spaces.Discrete(2)),
        (
            "xyz",
            spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        ),
        (
            "rpy",
            spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
        ),
        (
            "grasp",
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        ),
    ])
    if self.config['norm_actions'] == 'across_actions':
      action_space_dict['mean'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
      action_space_dict['std'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    action_space = spaces.Dict(action_space_dict)

    action_mean = None
    action_std = None
    if self.config['norm_actions'] == 'gaussian':
      action_std = self.action_tokenizer.action_std
      action_mean = self.action_tokenizer.action_mean
    self.model = TransformerNetwork(
        observation_history_length=self.config['observation_history_size'],
        future_prediction_length=self.config['future_action_window_size'],
        token_embedding_dim=self.config['token_embedding_dim'],
        causal_attention=self.config['causal_attention'],
        num_layers=self.config['num_layers'],
        layer_size=self.config['layer_size'],
        observation_space=observation_space,
        action_space=action_space,
        image_keys=['image_primary'],
        context_key='natural_language_embedding',
        action_mean=action_mean,
        action_std=action_std,
    )

  #@profile
  def training_step(self, batch, batch_idx):
    self.accuracy = self.accuracy.to(self.device)
    self.model = self.model.to(self.device)
    if isinstance(batch, list):
      instruction_strs = sum([batch[i]['language_instruction'] for i in range(len(batch))], [])
    else:
      instruction_strs = copy(batch['language_instruction'])

    video, instructions, action = load_batch(batch,
                                             self.window_size,
                                             self.total_window_size,
                                             embed_instructions=True)
    video = video.to(self.device)
    instructions = instructions.to(self.device)
    del batch

    obs = {
        'image_primary': video,
        'natural_language_embedding': repeat(instructions, 'b n -> b f n', f=video.shape[1])
    }

    network_state = np_to_tensor(
        batched_space_sampler(
            self.model.state_space,
            batch_size=video.shape[0],
        ))

    outs, _ = self.model(
        obs,
        network_state,
    )
    outs = outs[:, :(self.future_action_window_size + 1), :, :]

    ground_truth = self.action_tokenizer.tokenize_to_xyzrpyg(
        action, self.config['norm_actions']).to(self.device)
    loss = self.criterion(rearrange(outs, 'b f a bins -> (b f a) bins'),
                          rearrange(ground_truth, 'b f a -> (b f a)'))

    out_preds = torch.argmax(outs, -1)
    acc = self.accuracy(out_preds, ground_truth)
    log_metrics(
        self,
        loss.item(),
        acc,
        out_preds.cpu().detach(),
        ground_truth.cpu().detach(),
        video.cpu().detach() if self.log_images else None,
        instructions=instruction_strs,
        batch_idx=batch_idx,
    )
    del out_preds, ground_truth, video, instructions, action
    return loss


def oxe_data_module(config: dict) -> LightningDataModule:
  new_config = config
  config = default_config()
  if new_config is not None:
    config.update(new_config)
  if config['checkpoint_path'] is not None and config['resume'] and  config['resume'] != 'all':
    return CombinedOXEHD5DataModule.load_from_checkpoint(config['checkpoint_path'])

  return CombinedOXEHD5DataModule(
      batch_size=config['batch_size'],
      datapaths=config['local_datasets'],
      shuffle=config['shuffle'],
      seed=config['seed'],
      embed_instructions=config['model'] == 'rt1',
      oxe=config['oxe'],
      oxe_batch_percentage=config['oxe_batch_percentage'],
      oxe_datasets=config['oxe_datasets'],
      shuffle_buffer_size=config['shuffle_buffer_size'],
      data_augmentation=config['data_augmentation'],
      random_erasing=config['random_erasing'],
      future_action_window_size=config['future_action_window_size'],
      num_parallel_calls=config['num_parallel_calls'],
      num_threads=config['num_threads'],
  )


def oxe_module(config: dict):
  new_config = config
  config = default_config()
  if new_config is not None:
    config.update(new_config)
  if config['model'] == 'rt1':
    module = RT1Module
  else:
    module = RTX1Module

  if config['checkpoint_path'] and config['resume'] != 'all':
    module = module.load_from_checkpoint(
        config['checkpoint_path'], map_location='cuda' if torch.cuda.is_available() else 'cpu')
  else:
    module = module(config)
  return module


def oxe_trainer(config: dict, wandb_logger: Optional[WandbLogger] = None, callbacks=None):
  new_config = config
  precision = config['precision']
  config = default_config()
  if new_config is not None:
    config.update(new_config)
  torch.set_float32_matmul_precision(config['matmul_precision'])
  if config['strategy'] == 'fsdp':
    strategy = FSDPStrategy(sharding_strategy=config['sharding_strategy'])
  elif 'ddp' in config['strategy']:
    strategy = DDPStrategy(gradient_as_bucket_view=True,
                           static_graph=True,
                           find_unused_parameters=('find_unused_parameters' in config['strategy']),
                           start_method='spawn' if 'spawn' in config['strategy'] else 'popen')
  else:
    strategy = config['strategy']

  if config['profile']:
    trainer = Trainer(profiler='advanced',
                      fast_dev_run=True,
                      strategy=strategy,
                      gradient_clip_val=config['gradient_clip_val'],
                      gradient_clip_algorithm='value')
  else:
    trainer = Trainer(strategy=strategy,
                      max_steps=config['max_steps'],
                      logger=wandb_logger,
                      callbacks=callbacks,
                      precision=precision,
                      accelerator='cpu' if config['cpu'] else 'auto',
                      log_every_n_steps=config['log_frequency'],
                      gradient_clip_val=config['gradient_clip_val'],
                      gradient_clip_algorithm='value',
                      accumulate_grad_batches=config['accumulate_grad_batches'],
                      overfit_batches=config['overfit_batches'])
  return trainer


def run_training(config: dict,
                 module: Optional[LightningModule] = None,
                 datamodule: Optional[LightningDataModule] = None,
                 trainer: Optional[Trainer] = None):
  wandb.login()
  wandb_logger = WandbLogger(project=config['project_name'], job_type='train')

  checkpoint_callback = ModelCheckpoint(
      filename=config['project_name'] + "_{step:02d}",
      dirpath=f"checkpoints/{config['model']}",
      every_n_train_steps=config['checkpoint_frequency'],
      save_top_k=-1,
  )

  summary_callback = ModelSummary(max_depth=config['summary_depth'])
  if module is None:
    module = oxe_module(config)
  if datamodule is None:
    datamodule = oxe_data_module(config)
  if trainer is None:
    trainer = oxe_trainer(config, wandb_logger, [checkpoint_callback, summary_callback])

  if config['reload_every_n_steps'] <= 0:
    trainer.fit(module,
                datamodule=datamodule,
                ckpt_path=config['checkpoint_path'] if config['resume'] == 'all' else None)

  else:
    num_iterations = config['skip_iterations']
    for i in range(int(config['max_steps'] // config['reload_every_n_steps'] + 1)):
      config['max_steps'] = config['reload_every_n_steps'] + num_iterations
      trainer = oxe_trainer(config, wandb_logger, [checkpoint_callback, summary_callback])
      trainer.fit(
          module,
          datamodule=datamodule,
          ckpt_path=config['checkpoint_path'] if config['resume'] != 'all' and i == 0 else None)
      num_iterations += config['reload_every_n_steps']
      module.config['skip_iterations'] = num_iterations
      module.num_reloads += 1


def default_config():
  abls_flags = FLAGS.get_flags_for_module(__name__)
  return {abs_flag.name: abs_flag.value for abs_flag in abls_flags}


def main(_):
  config = default_config()
  run_training(config)


if __name__ == "__main__":
  app.run(main)
