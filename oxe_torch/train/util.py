import torch
from pytorch_lightning import LightningModule
import numpy as np
# from pytorch_lightning.loggers import WandbLogger
# from pandas import DataFrame
def log_metrics(module: LightningModule,
                loss,
                acc,
                out_preds,
                ground_truth,
                video=None,
                instructions=None,
                batch_idx=None,
 ):
    
    log_idx = int((batch_idx / module.log_frequency)) % out_preds.shape[0] if batch_idx is not None else 0
    x_idx = 0
    y_idx = 1
    z_idx = 2
    roll_idx = 3
    pitch_idx = 4
    yaw_idx = 5
    grasp_idx = 6
    with torch.no_grad():
        out_preds = out_preds.float()
        ground_truth = ground_truth.float()
        metrics = {
            'x_pred0_train': out_preds[log_idx, 0, x_idx],
            'x_gt0_train': ground_truth[log_idx, 0, x_idx],
            'y_pred0_train': out_preds[log_idx, 0, y_idx],
            'y_gt0_train': ground_truth[log_idx, 0, y_idx],
            'z_pred0_train': out_preds[log_idx, 0, z_idx],
            'z_gt0_train': ground_truth[log_idx, 0, z_idx],
            'roll_pred0_train': out_preds[log_idx, 0, roll_idx],
            'roll_gt0_train': ground_truth[log_idx, 0, roll_idx],
            'pitch_pred0_train': out_preds[log_idx, 0, pitch_idx],
            'pitch_gt0_train': ground_truth[log_idx, 0, pitch_idx],

            'yaw_pred0_train': out_preds[log_idx, 0, yaw_idx],
            'yaw_gt0_train': ground_truth[log_idx, 0, yaw_idx],
            'grasp_pred0_train': out_preds[log_idx, 0, grasp_idx],
            'grasp_gt0_train': ground_truth[log_idx, 0, grasp_idx],

            'train_step': module.global_step,
            'epoch': module.current_epoch,
            'loss': loss,
            'acc': acc,
            'lr': module.trainer.optimizers[0].param_groups[0]['lr'],
        }
        if module.future_action_window_size > 0:
            last_idx = module.future_action_window_size
            metrics.update({ 
                f'x_pred{last_idx}_train': out_preds[log_idx, last_idx, x_idx],
                f'x_gt{last_idx}_train': ground_truth[log_idx, last_idx, x_idx],
            })
        # if instructions is not None:
        #     wandb_logger: WandbLogger  = module.logger.experiment
        #     wandb_logger.log_text('text', (DataFrame(instructions)))
        if batch_idx is not None:
            metrics['batch_idx'] = batch_idx

        module.log_dict(metrics, on_step=True, prog_bar=False)
        module.log_dict(
            {
                'loss_': loss,
                'acc_': acc,
                'lr_': module.trainer.optimizers[0].param_groups[0]['lr'],
                'train_step_': module.global_step,
            },
            on_step=True,
            prog_bar=True)

        if video is not None and batch_idx % module.log_image_frequency == 0:
            instructions = instructions if instructions is not None else [
            '' for _ in range(video.shape[0])
        ]
            captions = [
                f'batch_idx={log_idx}, {instructions[log_idx]} gt: {str(list(ground_truth[log_idx,0,:].tolist()))}, pred: {str(list((out_preds[log_idx,0,:].tolist())))}'
            
            ]
            module.logger.log_video('video', [
                np.array(255 * video[log_idx, :, :, :, :].detach().to('cpu')).astype(np.uint8),
            ],
                                    caption=captions)
