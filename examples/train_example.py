from absl import flags
import torch
from torch import nn, optim
import tqdm
from torch.utils.data import DataLoader
from oxe_torch.data.dataset import get_oxe_dataset, TorchRLDSDataset
from robo_transformers.inference_server import InferenceServer
from tensorboardX import SummaryWriter
import os
import tensorflow as tf
import pytorch_warmup as warmup
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from einops import rearrange, reduce
from oxe_torch.action_tokenization import RTX1ActionTokenizer
from oxe_torch.train.dist import init_distributed, is_main_process, get_rank, get_world_size, is_dist_avail_and_initialized
import wandb
import numpy as np

tf.config.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_integer("num_warmup_steps", 1000, "Number of warmup steps.")
flags.DEFINE_integer("shuffle_buffer_size", 1000, "Shuffle buffer size.")
flags.DEFINE_integer("eval_batch_size", 1, "Eval Batch size.")
flags.DEFINE_float("lr", 1e-4, "Learning Rate.")
flags.DEFINE_float("min_lr", 1e-6, "Min Learning Rate.")
flags.DEFINE_float("weight_decay", 0, "Weight Decay.")
flags.DEFINE_string("dataset_name", "fractal20220817_data", "Dataset name.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory.")
flags.DEFINE_list("baselines", [], "Baselines to evaluate against.")
flags.DEFINE_bool("data_augmentation", True,
                  "Whether or not to use data augmentation.")
flags.DEFINE_float("conditioning_scale", 1.0,
                   "Scale of film conditioning. on text input.")
flags.DEFINE_float("label_smoothing", 0.0, "Label smoothing.")
flags.DEFINE_string("loss", "cse", "Loss function.")
flags.DEFINE_bool("freeze_vit", False, "Freeze ViT weights.")


def eval(model: torch.nn.Module,
         action_tokenizer: RTX1ActionTokenizer,
         writer: SummaryWriter,
         step_num,
         eval_data_loader,
         criterion,
         device,
         baseline_keys=[],
         conditioning_scale=1.0):
    # evaluate
    print('evaluating')
    model.eval()
    mse = nn.MSELoss()
    with torch.no_grad():
        eval_loss = 0
        future_eval_loss = 0
        eval_acc = 0
        future_eval_acc = 0
        baselines = {}

        for baseline in baseline_keys:
            baselines[baseline] = {
                'loss':
                    0,
                'acc':
                    0,
                'mse':
                    0,
                'model':
                    InferenceServer(
                        baseline.split('/')[0],
                        baseline.split('/')[1])
            }

        eval_steps = 0.
        for _, sample in tqdm.tqdm(enumerate(eval_data_loader)):
            if (eval_steps == 1):
                break
            video = rearrange(sample['observation']['image_primary'] / 255.,
                              'b f h w c -> b f c h w').to(device)
            instructions = sample['language_instruction']
            ground_truth = action_tokenizer.tokenize_xyzrpyg(
                sample['action'], device)[:,-1,:]

            outs = reduce(model.run(video, instructions, conditioning_scale), 'b f a bins -> b a bins', 'mean')
            out_preds = torch.max(outs, -1)[1]

            eval_loss += criterion(
                rearrange(outs, 'b a bins -> (b a) bins'),
                rearrange(ground_truth, 'b a -> (b a)')).detach().to('cpu')
            eval_acc += (
                out_preds == ground_truth).float().mean().detach().to('cpu')

            future_out_one_hot = nn.functional.one_hot(out_preds,
                                                       256).to(device).float()
            # future_gt = ground_truth[:, -1, :]
            future_gt_raw = sample['action'][:, -1, :]

            # future_eval_loss += criterion(
            #     rearrange(future_out_one_hot, 'b a bins -> (b a) bins'),
            #     rearrange(future_gt, 'b a -> (b a)')).detach().to('cpu')
            # future_eval_acc += (
            #     out_preds[:,
            #               -1, :] == future_gt).float().mean().detach().to('cpu')

            batch_size = video.shape[0]
            n_frames = video.shape[1]

            # Log imagea and action frames in batch first sample:
            # for i in range(n_frames):
            #     writer.add_image('image',
            #                      video[0, i, :, :, :],
            #                      step_num + n_frames * eval_steps + i,
            #                      dataformats='CHW')
            #     writer.add_text('instruction', instructions[0],
            #                     step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('x_gt', ground_truth[0, i, 8],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('y_gt', ground_truth[0, i, 9],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('z_gt', ground_truth[0, i, 10],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('roll_gt', ground_truth[0, i, 4],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('pitch_gt', ground_truth[0, i, 5],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('yaw_gt', ground_truth[0, i, 6],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('grasp_gt', ground_truth[0, i, 3],
            #                       step_num + n_frames * eval_steps + i)

            #     writer.add_scalar('x_pred', out_preds[0, i, 8],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('y_pred', out_preds[0, i, 9],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('z_pred', out_preds[0, i, 10],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('roll_pred', out_preds[0, i, 4],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('pitch_pred', out_preds[0, i, 5],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('yaw_pred', out_preds[0, i, 6],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('grasp_pred', out_preds[0, i, 3],
            #                       step_num + n_frames * eval_steps + i)

            #     writer.add_scalar('x_gt_raw', sample['action'][0, i, 0],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('y_gt_raw', sample['action'][0, i, 1],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('z_gt_raw', sample['action'][0, i, 2],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('roll_gt_raw', sample['action'][0, i, 3],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('pitch_gt_raw', sample['action'][0, i, 4],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('yaw_gt_raw', sample['action'][0, i, 5],
            #                       step_num + n_frames * eval_steps + i)
            #     writer.add_scalar('grasp_gt_raw', sample['action'][0, i, 6],
            #                       step_num + n_frames * eval_steps + i)

            wandb.log(
            {
                'image_frames':
                    wandb.Video(
                        np.array(255 * video[0, :, :, :, :].detach().to(
                            'cpu')).astype(np.uint8),
                        caption=
                        f" gt: {str(ground_truth[0,:])}, pred: {str(out_preds[0,:])}"
                    ),
                # 'instruction': instructions[0],
                'train_step': step_num
            })
            wandb.log({
                'x_gt': ground_truth[0,  8],
                'y_gt': ground_truth[0,  9],
                'z_gt': ground_truth[0,  10],
                'roll_gt': ground_truth[0,  4],
                'pitch_gt': ground_truth[0,  5],
                'yaw_gt': ground_truth[0,  6],
                'grasp_gt': ground_truth[0,  3],
                'train_step': step_num
            })
            wandb.log({
                'x_pred': out_preds[0, 8],
                'y_pred': out_preds[0, 9],
                'z_pred': out_preds[0, 10],
                'roll_pred': out_preds[0, 4],
                'pitch_pred': out_preds[0, 5],
                'yaw_pred': out_preds[0, 6],
                'grasp_pred': out_preds[0, 3],
                'train_step': step_num
            })
            wandb.log({
                'x_gt_raw': sample['action'][0, -1, 0],
                'y_gt_raw': sample['action'][0, -1, 1],
                'z_gt_raw': sample['action'][0, -1, 2],
                'roll_gt_raw': sample['action'][0, -1, 3],
                'pitch_gt_raw': sample['action'][0, -1, 4],
                'yaw_gt_raw': sample['action'][0, -1, 5],
                'grasp_gt_raw': sample['action'][0, -1, 6],
                'train_step': step_num,
                'instruction': instructions[0]
            })

            video = rearrange(video, 'b f c h w -> b f h w c') * 255
            for baseline in FLAGS.baselines:
                baseline_model = baselines[baseline]['model']
                batch_actions = torch.zeros((video.shape[0], 11, 256),
                                            dtype=torch.float32,
                                            device=device)
                batch_actions_raw = torch.zeros((video.shape[0], 7),
                                                dtype=torch.float32,
                                                device='cpu')
                for i in range(batch_size):
                    for j in range(n_frames):
                        out_raw = baseline_model(image=(video[i, j, :, :, :] ).cpu().numpy(),
                                                 instruction=instructions[i],
                                                 save=False)
                        batch_actions_raw[i, :] = torch.tensor([
                            out_raw['world_vector'][0],
                            out_raw['world_vector'][1],
                            out_raw['world_vector'][2],
                            out_raw['rotation_delta'][0],
                            out_raw['rotation_delta'][1],
                            out_raw['rotation_delta'][2],
                            out_raw['gripper_closedness_action']
                        ],
                                                               device='cpu')
                        # print(f' \n\n   {baseline} out',out)
                        out = action_tokenizer.tokenize_dict(out_raw, device)
                        batch_actions[i, :, :] = nn.functional.one_hot(
                            out, 256).to(device)

                        # Log action frames in batch first sample:
                        if i == 0:
                            baseline_name = baseline.replace('/', '_').replace(
                                '-', '_')
                            writer.add_scalar(
                                'x_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out[8], step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'y_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out[9], step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'z_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out[10], step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'roll_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out[4], step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'pitch_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out[5], step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'yaw_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out[6], step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'grasp_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out[3], step_num + n_frames * eval_steps + j)

                            writer.add_scalar(
                                'x_raw_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out_raw['world_vector'][0],
                                step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'y_raw_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out_raw['world_vector'][1],
                                step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'z_raw_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out_raw['world_vector'][2],
                                step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'roll_raw_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out_raw['rotation_delta'][0],
                                step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'pitch_raw_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out_raw['rotation_delta'][1],
                                step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'yaw_raw_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out_raw['rotation_delta'][2],
                                step_num + n_frames * eval_steps + j)
                            writer.add_scalar(
                                'grasp_raw_' +
                                baseline.replace('/', '_').replace('-', '_'),
                                out_raw['gripper_closedness_action'],
                                step_num + n_frames * eval_steps + j)

                            if j == 5:
                                wandb.log(
                                    {
                                        'x_' + baseline_name: out[8],
                                        'y_' + baseline_name: out[9],
                                        'z_' + baseline_name: out[10],
                                        'roll_' + baseline_name: out[4],
                                        'pitch_' + baseline_name: out[5],
                                        'yaw_' + baseline_name: out[6],
                                        'grasp_' + baseline_name: out[3],
                                        'train_step': step_num
                                    })
               
                                wandb.log(
                                    {
                                        'x_raw_' + baseline_name:
                                            out_raw['world_vector'][0],
                                        'y_raw_' + baseline_name:
                                            out_raw['world_vector'][1],
                                        'z_raw_' + baseline_name:
                                            out_raw['world_vector'][2],
                                        'roll_raw_' + baseline_name:
                                            out_raw['rotation_delta'][0],
                                        'pitch_raw_' + baseline_name:
                                            out_raw['rotation_delta'][1],
                                        'yaw_raw_' + baseline_name:
                                            out_raw['rotation_delta'][2],
                                        'grasp_raw_' + baseline_name:
                                            out_raw['gripper_closedness_action'],
                                        'train_step': step_num
                                    })

                        # print(f' \n\n   {baseline} tokenized',out)

                # print(f' \n\n   {baseline} action', torch.max(batch_actions[-1,:,:],-1)[1])
                baselines[baseline]['loss'] += criterion(
                    batch_actions[0, :, :], ground_truth[0, :]).to('cpu').detach()
                baselines[baseline]['acc'] += (torch.max(
                    batch_actions[0, :, :], -1)[1] == ground_truth[
                        0, :]).float().mean().to('cpu').detach()

                baselines[baseline]['mse'] += mse(
                    batch_actions_raw[0, :],
                    future_gt_raw[0, :]).to('cpu').detach()

            eval_steps += 1

    writer.add_scalar('eval_loss', eval_loss / eval_steps, step_num)
    writer.add_scalar('future_eval_loss', future_eval_loss / eval_steps,
                      step_num)
    writer.add_scalar('eval_acc', eval_acc / eval_steps, step_num)
    writer.add_scalar('future_eval_acc', future_eval_acc / eval_steps, step_num)

    wandb.log({
        'eval_loss': eval_loss / eval_steps,
        'future_eval_loss': future_eval_loss / eval_steps,
        'eval_acc': eval_acc / eval_steps,
        'future_eval_acc': future_eval_acc / eval_steps,
        'train_step': step_num})

    for baseline in FLAGS.baselines:
        baseline_name = baseline.replace('/', '_').replace('-', '_')
        writer.add_scalar(
            f"{baseline.replace('/','_').replace('-','_')}_future_loss",
            baselines[baseline]['loss'] / eval_steps, step_num)
        writer.add_scalar(
            f"{baseline.replace('/','_').replace('-','_')}_future_acc",
            baselines[baseline]['acc'] / eval_steps, step_num)
        wandb.log({
            f"{baseline_name}_future_loss":
                baselines[baseline]['loss'] / eval_steps,
            f"{baseline_name}_future_acc":
                baselines[baseline]['acc'] / eval_steps,
            'train_step': step_num 
        })
        wandb.log({
            f"{baseline_name}_future_mse":
                baselines[baseline]['mse'] / eval_steps,
            'train_step': step_num
        })

    writer.flush()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(model: torch.nn.Module, action_tokenizer):
    """
    Runs the training loop.
    """
    init_distributed()
    if is_main_process():
        wandb.init(
            # set the wandb project where this run will be logged
            project="rtdiffusion",
            config=dict(num_epochs=FLAGS.num_epochs,
                        batch_size=FLAGS.batch_size,
                        num_warmup_steps=FLAGS.num_warmup_steps,
                        shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                        eval_batch_size=FLAGS.eval_batch_size,
                        lr=FLAGS.lr,
                        min_lr=FLAGS.min_lr,
                        weight_decay=FLAGS.weight_decay,
                        dataset_name=FLAGS.dataset_name,
                        checkpoint_dir=FLAGS.checkpoint_dir,
                        baselines=FLAGS.baselines,
                        data_augmentation=FLAGS.data_augmentation,
                        conditioning_scale=FLAGS.conditioning_scale,
                        label_smoothing=FLAGS.label_smoothing,
                        loss=FLAGS.loss,
                        freeze_vit=FLAGS.freeze_vit))
    conditioning_scale = FLAGS.conditioning_scale
    writer = None
    if is_main_process():
        writer = SummaryWriter()
    train_ds = TorchRLDSDataset(*get_oxe_dataset(
        FLAGS.dataset_name,
        train=True,
        data_augmentation=FLAGS.data_augmentation,
        shuffle_buffer_size=FLAGS.shuffle_buffer_size),
                                train=True,
                                rank=get_rank(),
                                world_size=get_world_size())
    eval_ds = None
    if is_main_process():
        eval_ds = TorchRLDSDataset(*get_oxe_dataset(
            FLAGS.dataset_name,
            train=False,
            data_augmentation=False,
            shuffle_buffer_size=FLAGS.shuffle_buffer_size),
                                   train=False,
                                   rank=0,
                                   world_size=1)

    train_data_loader = DataLoader(
        train_ds,
        batch_size=FLAGS.batch_size,
        num_workers=
        0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
        pin_memory=True,
        # sampler= DistributedSampler(dataset=train_ds, shuffle=True) if torch.cuda.device_count() > 1 else None
    )
    eval_data_loader = None
    # if is_main_process():
    #     eval_data_loader = DataLoader(
    #         eval_ds,
    #         batch_size=FLAGS.eval_batch_size,
    #         num_workers=
    #         0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    #         pin_memory=True,
    #         # shuffle=True,
    #         # sampler= DistributedSampler(dataset=eval_ds, shuffle=False) if torch.cuda.device_count() > 1 else None
    #     )

    # steps_per_epoch = len(train_data_loader)
    warmup_period = FLAGS.num_warmup_steps
    # num_steps = steps_per_epoch * FLAGS.num_epochs - warmup_period
    t0 = 250 * 5
    lr_min = FLAGS.min_lr
    max_step = t0 * 50 + warmup_period

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if FLAGS.freeze_vit:
        for param in model.vit.parameters():
            param.requires_grad = False
    if is_main_process():
        print('\n\n Training model with {} parameters'.format(
            count_parameters(model)))

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if is_main_process():
            print(f'Using {torch.cuda.device_count()} GPUs')
        # Convert BatchNorm to SyncBatchNorm.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DDP(model,
                    device_ids=[get_rank()],
                    output_device=get_rank(),
                    find_unused_parameters=True)

        model.run = model.module.run
        model.train_step = model.module.train_step
        if is_main_process():
            wandb.watch(model.module, log_freq=100)

    criterion = nn.MSELoss() if FLAGS.loss == 'mse' else nn.CrossEntropyLoss()
    if is_dist_avail_and_initialized():
        optimizer = ZeroRedundancyOptimizer(model.parameters(),
                                            optimizer_class=torch.optim.AdamW,
                                            lr=FLAGS.lr,
                                            weight_decay=FLAGS.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(),
                                lr=FLAGS.lr,
                                weight_decay=FLAGS.weight_decay)
    # optimizer.zero_grad()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t0, T_mult=2, eta_min=lr_min)

    warmup_scheduler = warmup.LinearWarmup(optimizer,
                                           warmup_period=warmup_period)
    step_num = 0

    # lr_finder = LRFinder(model, optimizer, criterion, device=device)
    # lr_finder.range_test(train_data_loader, start_lr=1e-7, end_lr=10, num_iter=100)
    # ax, lr = lr_finder.plot(suggest_lr=True) # to inspect the loss-learning rate graph
    # print('\n\n\n lr', lr)
    # lr_finder.reset() # to reset the model and optimizer to their initial state
    for epoch in range(FLAGS.num_epochs):
        # if torch.cuda.device_count() > 1:
        #     train_data_loader.sampler.set_epoch(epoch)
        if is_main_process():
            print(f'epoch {epoch}')
            wandb.log({'epoch': epoch})
        dist.barrier()
        for i, sample in tqdm.tqdm(enumerate(train_data_loader)):

            # if step_num % 10000 == 0:
            #     if is_main_process():
            #         eval(model, action_tokenizer, writer, step_num,
            #              eval_data_loader, criterion, device, FLAGS.baselines,
            #              conditioning_scale)
            #     if torch.cuda.device_count() > 1:
            #         dist.barrier()
            # if i == 250:
            #     break

            video = rearrange(sample['observation']['image_primary'] / 255.0,
                              'b f h w c -> b f c h w').to(device)
            instructions = sample['language_instruction']
            ground_truth = action_tokenizer.tokenize_xyzrpyg(
                sample['action'], device)[:,-1,:]

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():

            outs = model.train_step(video, instructions)[:,-1,:,:] # only take last frame
            out_preds = torch.max(outs, -1)[1]

            loss = criterion(rearrange(outs, 'b a bins -> (b a) bins'),
                             rearrange(ground_truth, 'b a -> (b a)'))
            loss.backward()
            optimizer.step()
            acc = (out_preds == ground_truth).float().mean().detach().to('cpu')

            if is_main_process():
                writer.add_scalar('loss',
                                  float(loss.to('cpu').detach().numpy()),
                                  step_num)
                writer.add_scalar('acc', float(acc.to('cpu').detach().numpy()),
                                  step_num)
                wandb.log(
                    {
                        'loss': float(loss.to('cpu').detach().numpy()),
                        'acc': float(acc.to('cpu').detach().numpy()),
                        'lr': optimizer.param_groups[0]['lr'],
                        'x_pred_train': out_preds[0, 8],
                        'x_gt_train': ground_truth[0, 8],
                        'y_pred_train': out_preds[0, 9],
                        'y_gt_train': ground_truth[0, 9],
                        'z_pred_train': out_preds[0, 10],
                        'z_gt_train': ground_truth[0, 10],
                        'roll_pred_train': out_preds[0, 4],
                        'roll_gt_train': ground_truth[0, 4],
                        'pitch_pred_train': out_preds[0, 5],
                        'pitch_gt_train': ground_truth[0, 5],
                        # 'instruction_train': instructions[0],
                        'batch_idx': i,
                        'train_step': step_num
                    })
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    lr_scheduler.step()
            if warmup_scheduler.last_step + 1 >= max_step:
                break
            if is_main_process():
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                  step_num)

            if (step_num + 1) % 1000 == 0:
                # save model
                if is_main_process():
                    os.makedirs(
                        f'{FLAGS.checkpoint_dir}/{FLAGS.model}_{FLAGS.dataset_name}',
                        exist_ok=True)
                    if is_dist_avail_and_initialized():
                        torch.save(
                            model.module.state_dict(),
                            f'{FLAGS.checkpoint_dir}/{FLAGS.model}_{FLAGS.dataset_name}/step{step_num}.pt'
                        )
                    else:
                        torch.save(
                            model.state_dict(),
                            f'{FLAGS.checkpoint_dir}/{FLAGS.model}_{FLAGS.dataset_name}/step{step_num}.pt'
                        )
            step_num += 1
