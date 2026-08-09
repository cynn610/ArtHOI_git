import numpy as np
from natsort import natsorted
import torch
from loguru import logger as loguru

"""
three params:
mean betas of shape [10]
hand_trans of shape [frames, 3]
hand_poses of shape [frames, 3+45] = (global_orient/rotation:=3 + joints:=45)
"""

def dump_params(mano_params : dict, frames_cnt):
    assert 'betas' in mano_params, f"missing 'betas' in mano_params"
    assert 'transl' in mano_params, f"missing 'transl' in mano_params"
    assert 'hand_pose' in mano_params, f"missing 'hand_pose' in mano_params"
    assert 'global_orient' in mano_params, f"missing 'global_orient' in mano_params"
    
    mean_betas = mano_params['betas']
    if len(mean_betas.shape) == 2 and mean_betas.shape[0] == frames_cnt:
        mean_betas = mean_betas.mean(axis=0)
    assert mean_betas.shape == (10,)
    
    hand_trans = mano_params['transl']
    assert hand_trans.shape == (frames_cnt, 3)
    
    hand_poses = mano_params['hand_pose']
    hand_rot = mano_params['global_orient']
    assert hand_rot.shape == (frames_cnt, 3)
    assert hand_poses.shape == (frames_cnt, 45)
    hand_poses = torch.cat([hand_rot, hand_poses], dim=1)
    assert hand_poses.shape == (frames_cnt, 48)
    
    loguru.info(f'mean betas: {mean_betas}\nmin: {mean_betas.min()}, max: {mean_betas.max()}\nsum: {mean_betas.sum()}')
    
    return {
        'hand_poses': hand_poses.detach().cpu().numpy(),  # [frames, 48]
        'hand_trans': hand_trans.detach().cpu().numpy(),  # [frames, 3]
        'mean_shape': mean_betas.detach().cpu().numpy(),  # [10]
    }

def dump_instance_params(hand_params, out_path, frames_cnt):
    """Dump arbitrary hand instances while preserving MANO handedness."""
    payload = {}
    for instance_id, mano_param in hand_params.items():
        if mano_param is None:
            continue
        if hasattr(mano_param, "get_i_dict"):
            params = mano_param.get_i_dict()
            handedness = mano_param.handedness
        else:
            params = mano_param["params"]
            handedness = mano_param["handedness"]
        entry = dump_params(params, frames_cnt)
        entry["handedness"] = handedness
        entry["flat_hand_mean"] = bool(getattr(mano_param, "flat_hand_mean", False))
        payload[instance_id] = entry
    np.save(f'{out_path}/data_hand.npy', payload, allow_pickle=True)

def dump_LR_params(mano_params_r, mano_params_l, out_path, frames_cnt):
    if mano_params_l is not None:
        l_dict = dump_params(mano_params_l, frames_cnt)
    else:
        loguru.warning("Left hand parameters are None, skipping left hand dump.")
        l_dict = None
    if mano_params_r is not None:
        r_dict = dump_params(mano_params_r, frames_cnt)
    else:
        loguru.warning("Right hand parameters are None, skipping right hand dump.")
        r_dict = None
    
    _d = {'right': r_dict, 'left': l_dict}
    np.save(f'{out_path}/data_hand.npy', _d, allow_pickle=True)
