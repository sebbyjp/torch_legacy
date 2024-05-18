from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.schedulers import ASHAScheduler
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import Trainer
from ray.train.torch import TorchTrainer
from oxe_torch.train.oxe_training_module import oxe_module, oxe_data_module
from ray import tune
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_tuners', 4, 'Number of tuners to run in parallel')
flags.DEFINE_integer('num_tune_samples', 3, 'Number of samples to run for each tuner')

def find_lr(trainer, module, datamodule):
  tuner = Tuner(trainer)
  lr_finder = tuner.lr_find(module,
                            train_dataloaders=datamodule.train_dataloader(),
                            num_training=20,
                            max_lr=1e-1,
                            min_lr=1e-6,
                            mode="exponential",
                            early_stop_threshold=None)
  # Results can be found in
  print(lr_finder.results)
  # Plot with
  fig = lr_finder.plot(suggest=True)
  fig.savefig("lr_finder.png")
  # Pick point based on plot, or get suggestion
  new_lr = lr_finder.suggestion()
  return new_lr

def tune_oxe(num_samples=3, num_steps=10, module_config=None):
  if module_config['cpu']:
    resources_per_worker = {"CPU": 1}
    use_gpu = False
  else:
    resources_per_worker = {"GPU": 1, "CPU": module_config['num_threads']}
    use_gpu = True
  scaling_config = ScalingConfig(num_workers=FLAGS.num_tuners,
                               use_gpu=use_gpu,
                               resources_per_worker=resources_per_worker,
                              )
  run_config = RunConfig(checkpoint_config=CheckpointConfig(
      num_to_keep=2,
      checkpoint_score_attribute="loss",
      checkpoint_score_order="min",
  ),)
  def train_func(config):
    dm = oxe_data_module(config)
    model = oxe_module(config)
    trainer = Trainer(
        devices="auto",
        accelerator='cpu' if module_config['cpu'] else 'auto',
        strategy=RayDDPStrategy(find_unused_parameters=True),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=True,
        log_every_n_steps=module_config['log_frequency'],
        precision=module_config['precision'],
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)

  # Define a TorchTrainer without hyper-parameters for Tuner
  ray_trainer = TorchTrainer(
      train_func,
      run_config=run_config,
      scaling_config=scaling_config,
  )
  search_space = {
      # "weight_decay":tune.choice([1e-1, 1e-2,1e-3]),
      # "image_tokens_size": tune.choice([8, 16]),
      # "num_layers": tune.choice([1, 8]),
      # "layer_size": tune.choice([128, 512]),
      "lr": tune.choice([1e-5, 3e-4]),
      # "gradient_clip_val": tune.choice([0.5, 10.0, 100.0]),
      "action_norm": tune.choice(["gaussian", "across_actions", None]),
      "causal_attention": tune.choice([True, False]),
      "bias": tune.choice([True, False]),


  }
  scheduler = ASHAScheduler(time_attr='total_time_s',max_t=num_steps, grace_period=1, reduction_factor=2)
  tuner = tune.Tuner(
      ray_trainer,
      param_space={"train_loop_config": search_space},
      tune_config=tune.TuneConfig(
          metric="loss",
          mode="min",
          num_samples=FLAGS.num_tune_samples,
          reuse_actors=True,
          scheduler=scheduler,
      ),
  )
  return tuner.fit()