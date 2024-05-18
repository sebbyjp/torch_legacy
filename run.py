
from absl import app, flags, logging
from oxe_torch.train.oxe_training_module import oxe_module
from oxe_torch.train.oxe_training_module import oxe_data_module
from oxe_torch.train.oxe_training_module import oxe_trainer
from oxe_torch.train.oxe_training_module import run_training
from oxe_torch.train.oxe_training_module import default_config

from oxe_torch.train.hparam_search import find_lr, tune_oxe
from lightning.pytorch import Trainer
import torch

FLAGS = flags.FLAGS
flags.DEFINE_bool('save_weights_and_exit', False, 'Save weights and exit')
flags.DEFINE_bool('find_lr', False, 'Find learning rate')
flags.DEFINE_bool('tune', False, 'Tune hyperparameters')

def main(_):
  config = default_config()

  datamodule = oxe_data_module(config)
  module = oxe_module(config)
  trainer: Trainer = oxe_trainer(config)
  if FLAGS.save_weights_and_exit:
      torch.save(module.model.state_dict(), 'weights.pt')
      'Weights saved to weights.pt'
      return
  
  if FLAGS.find_lr:
      new_lr = find_lr(trainer , module, datamodule)
      logging.info(f'New learning rate: {new_lr}')
      return
  if FLAGS.tune:
    results = tune_oxe(num_samples=5, num_steps=10, module_config=config)
    results.get_best_result(metric="loss", mode="min")
    print(results)
  run_training(config, module, datamodule)

if __name__ == "__main__":
  app.run(main)
