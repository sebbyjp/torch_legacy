import torch
from einops import repeat


class NormType:
  '''Normalization type'''
  NONE = 'none'
  GAUSSIAN = 'gaussian'
  ACROSS_ACTIONS = 'across_actions'


RTX1_ACTION_BOUNDS = {
    'base_displacement_vector': [-1.0, 1.0],
    'base_displacement_vertical_rotation': [-3.14159, 3.14159],
    'gripper_closedness_action': [0, 1.0],
    'rotation_delta': [-1.5708, 1.5708],
    'terminate_episode': [0.0, 1.0],
    'world_vector': [-1.0, 1.0]
}

RTX1_ACTION_VOCAB_SIZE = 256

# EEF POSE SMALL
# action mean= [ 0.01171502 -0.00441062  0.01226756  0.00941615 -0.00525671 -0.01514976
#   1.05081812], action std= [0.22805409 0.26354007 0.25224045 0.15780293 0.14854953 0.25473428
#  0.96879203]
ACTION_MEAN = [0.01171502, -0.00441062, 0.01226756, 0.00941615, -0.00525671, -0.01514976, 1.05081812]
ACTION_STD = [0.22805409, 0.26354007, 0.25224045, 0.15780293, 0.14854953, 0.25473428, 0.96879203]

# # Fractal
# ACTION_MEAN = [0.00174686, 0.00156649, -0.00315627, 0.01083297, -0.00143904, 0.00022827, 0.13385512]
# ACTION_STD = [0.01730303, 0.01491358, 0.01838284, 0.03902516, 0.0329105, 0.03648309, 0.12427615]



class RTX1ActionTokenizer:

  def __init__(self, bounds: dict = RTX1_ACTION_BOUNDS, vocab_size: int = RTX1_ACTION_VOCAB_SIZE):
    self.bounds = bounds
    self.vocab_size = vocab_size
    self.action_mean = torch.as_tensor(ACTION_MEAN)
    self.action_std = torch.as_tensor(ACTION_STD)

  def tokenize(self, action: torch.float, lower_bound: float, upper_bound: float) -> torch.long:
    action = torch.clip(action, lower_bound, upper_bound)
    action = (action - lower_bound) / (upper_bound - lower_bound)
    action = action * (self.vocab_size - 1)
    return action

  def detokenize(self, action: torch.long, lower_bound: float, upper_bound: float, mean=None, std=None) -> torch.float:
    action = action / (self.vocab_size - 1)
    action = (action * (upper_bound - lower_bound)) + lower_bound
    if mean is not None:
      action = (action * std) + mean
    return action


  def detokenize_vec(self, action_tokens: torch.long) -> torch.Tensor:

    action = torch.zeros(7, dtype=torch.float)
    action[6] = self.detokenize(action_tokens[3], self.bounds['gripper_closedness_action'][0],
                                self.bounds['gripper_closedness_action'][1])
    action[3:6] = self.detokenize(action_tokens[4:7], self.bounds['rotation_delta'][0],
                                  self.bounds['rotation_delta'][1])
    # tokens[7] = normalized_action['terminate_episode'][-1]
    action[0:3] = self.detokenize(action_tokens[8:], self.bounds['world_vector'][0],
                                  self.bounds['world_vector'][1])
    return action

#   def tokenize_dict(self, action: dict) -> torch.Tensor:
#     '''Converts a dict of float tensors to a 256-bit tensor.


#         Args:
#             action (dict): Input action:
#             {
#                 base_displacement_vector, list(Tensor((1,), torch.float64)) sz: 2
#                 base_displacement_vertical_rotation, list(Tensor((1,), torch.float64)) sz: 1
#                 gripper_closedness_action, list(Tensor((1,), torch.float64)) sz: 1
#                 rotation_delta, list(Tensor((1,), torch.float64)) sz: 3
#                 terminate_episode, list(Tensor((1,), torch.int64)) sz: 3
#                 world_vector, list(Tensor((1,), torch.float64)) sz: 3
#             }
#             bounds (dict): Action bounds. If None, default clipping from RT1 paper will be used:
#                 {
#                     'base_displacement_vector': [-1.0, 1.0],
#                     'base_displacement_vertical_rotation': [-3.14159, 3.14159],
#                     'gripper_closedness_action': [-1.0, 1.0],
#                     'rotation_delta': [-1.5708, 1.5708],
#                     'terminate_episode': [0, 1.0],
#                     'world_vector': [-1.0, 1.0]
#                 }
#             vocab_size (int): Vocabulary size. Default: 256.

#         Returns:
#             torch.Tensor: 11 dimentional int32 tensor in [0, vocab_size)
#         '''

#     normalized_action = {}
#     for k, v in action.items():
#       # normalized_action[k] = torch.concatenate(v).squeeze()
#       normalized_action[k] = self.tokenize(torch.as_tensor(v), self.bounds[k][0], self.bounds[k][1])

#     tokens = torch.ones(11, dtype=torch.long)
#     # tokens[0:2] = normalized_action['base_displacement_vector']
#     # tokens[2] = normalized_action['base_displacement_vertical_rotation']
#     tokens[3] = normalized_action['gripper_closedness_action']
#     tokens[4:7] = normalized_action['rotation_delta']
#     # tokens[7] = normalized_action['terminate_episode'][-1]
#     tokens[8:] = normalized_action['world_vector']
#     return tokens

#   def tokenize_xyzrpyg(self, action: torch.Tensor) -> torch.Tensor:
#     tokens = torch.ones((action.shape[0], action.shape[1], 11), dtype=torch.long)
#     tokens[:, :, 3] = self.tokenize(action[:, :, 6], self.bounds['gripper_closedness_action'][0],
#                                     self.bounds['gripper_closedness_action'][1])
#     tokens[:, :, 4:7] = self.tokenize(action[:, :, 3:6], self.bounds['rotation_delta'][0],
#                                       self.bounds['rotation_delta'][1])
#     tokens[:, :, 8:] = self.tokenize(action[:, :, 0:3], self.bounds['world_vector'][0],
#                                      self.bounds['world_vector'][1])
#     # print(tokens)
#     return tokens

  def normalize(self,
                action: torch.Tensor,
                norm_type: NormType = NormType.GAUSSIAN,
                mean=ACTION_MEAN,
                std=ACTION_STD) -> torch.Tensor:
    mean = torch.as_tensor(mean, device=action.device)
    std = torch.as_tensor(std, device=action.device)

    if norm_type is None or norm_type == NormType.NONE:
      return action
    
    elif norm_type == NormType.GAUSSIAN:
        action = (action - mean) / std
        return action


    elif norm_type == NormType.ACROSS_ACTIONS:
        mean = repeat(action.mean(dim=-1)  , 'b f -> b f a', a=7)
        std = repeat(action.std(dim=-1)  , 'b f -> b  f a', a=7)
        action = (action- mean) / std

        action = torch.concat([action, mean, std], dim=-1)
        return action


#   def unnormalize(self,
#                   action: torch.Tensor,
#                   norm_type: NormType = NormType.GAUSSIAN,
#                   mean=ACTION_MEAN,
#                   std=ACTION_STD) -> torch.Tensor:
#     mean = torch.as_tensor(mean, device=action.device)
#     std = torch.as_tensor(std, device=action.device)
#     if norm_type == NormType.NONE:
#       return action
#     elif norm_type == NormType.GAUSSIAN:
#       return (action * std) + mean
#     else:
#       if action.shape[-1] == 11:
#         mean = action[..., 0]
#         std = action[..., 1]
#         action[...,:2:] = action[...,:2:] * std + mean
#       else:
#         mean = action[..., -2]
#         std = action[..., -1]
#         action = ((action[...,:8] * std) + mean)

  def tokenize_to_xyzrpyg(self, action: torch.long, norm_type: NormType = NormType.NONE) -> torch.Tensor:
    action = self.normalize(action, norm_type)
    if norm_type == NormType.ACROSS_ACTIONS:
        tokens = torch.ones((action.shape[0], action.shape[1], 9), dtype=torch.long)
        tokens[...,7] = self.tokenize(action[...,3], -1, 1)
        tokens[...,8] = self.tokenize(action[...,6], -1, 1)
    else:
        tokens = torch.ones((action.shape[0], action.shape[1], 7), dtype=torch.long)
    tokens[:, :, :3] = self.tokenize(action[:, :, :3], self.bounds['world_vector'][0],
                                     self.bounds['world_vector'][1])
    tokens[:, :, 3:6] = self.tokenize(action[:, :, 3:6], self.bounds['rotation_delta'][0],
                                      self.bounds['rotation_delta'][1])
    tokens[:, :, 6] = self.tokenize(action[:, :, 6], self.bounds['gripper_closedness_action'][0],
                                    self.bounds['gripper_closedness_action'][1])

    # print(tokens.shape, tokens.dtype ,'\n\n\n')
    return tokens
    # print(tokens)
