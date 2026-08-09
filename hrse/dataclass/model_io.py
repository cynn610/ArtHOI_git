import os
import numpy as np
import torch
from datetime import datetime
from loguru import logger as loguru
from easydict import EasyDict as edict
import shutil
from arrgh import arrgh

# hack numpy for SMPLX
np.bool = np.bool_
np.int = np.int32

DEVICE = torch.device("cuda")

from hrse.dataclass.primitives import MANOParams, TransformParams, ArtiPart
from hrse.dataclass.dataset import ArtiDataset
from utils.data_util import to_numpy, to_tensor

def _handedness(instance_id, entry):
    handedness = entry.get('handedness', instance_id)
    if handedness not in ('left', 'right'):
        raise ValueError(
            f"Hand instance {instance_id!r} needs explicit left/right handedness"
        )
    return handedness

def load_parts_data(args, copy_to_output=True) -> \
        tuple[ArtiDataset, ArtiDataset, list[ArtiPart], list[list[TransformParams]]]:
    viseq = ArtiDataset(
        args.seq_path,
        use_video_inpainting=True,
        fix_intr=args.fix_intr, 
        fix_extr=args.fix_extr
    )
    viseq._to_device(DEVICE)
    nviseq = ArtiDataset(
        args.seq_path,
        use_video_inpainting=False,
        fix_intr=args.fix_intr, 
        fix_extr=args.fix_extr
    )
    nviseq._to_device(DEVICE)
    
    ## load part data
    parts = []
    obj = ArtiPart(None, None, None, DEVICE, name='obj')
    obj.subparts = []
    for i in range(args.part_cnt):
        part_data = np.load(
            f'{args.fit_ckpt}/part_{i}_partdata.npy', 
            allow_pickle=True
        ).item()
        part = ArtiPart(None, None, None, DEVICE, name=f'part_{i}')
        part.load_dict(part_data)
        obj.subparts.append(part)
        parts.append(part)
        if copy_to_output:
            shutil.copy(
                f'{args.fit_ckpt}/part_{i}_partdata.npy',
                f'{args.output_path}/part_{i}_partdata.npy'
            )
    allpart_params = []
    for part_id, part in enumerate(parts):
        param_dump = np.load(
            f'{args.fit_ckpt}/part_{part_id}_best.npy', 
            allow_pickle=True
        )
        part_param = [TransformParams(device=DEVICE, fit_scale=False, train=False) 
                        for _ in range(len(param_dump))]
        for i, p in enumerate(param_dump):
            part_param[i].set_param(p)
        allpart_params.append(part_param)
        if copy_to_output:
            shutil.copy(
                f'{args.fit_ckpt}/part_{part_id}_best.npy',
                f'{args.output_path}/part_{part_id}_best.npy'
            )
    loguru.critical(f'Loaded part params from {args.fit_ckpt} checkpoint path')
    return viseq, nviseq, parts, allpart_params

def load_ho_data(args) -> \
    tuple[ArtiDataset, ArtiDataset, list[ArtiPart], list[list[TransformParams]], dict[str, MANOParams]
]:
    """
    Load from args.fit_ckpt and args.seq_path
    ckpt contains:
        - part_{i}_partdata.npy: dict of part data
        - part_{i}_best.npy: list of TransformParams for each frame
        - data_hand.npy: dict of hand parameters for each side
    """
    viseq = ArtiDataset(
        args.seq_path,
        use_video_inpainting=True,
        fix_intr=args.fix_intr, 
        fix_extr=args.fix_extr
    )
    viseq._to_device(DEVICE)
    nviseq = ArtiDataset(
        args.seq_path,
        use_video_inpainting=False,
        fix_intr=args.fix_intr, 
        fix_extr=args.fix_extr
    )
    nviseq._to_device(DEVICE)
    
    F = nviseq.frame_cnt
    w2c = nviseq.w2c[0]
    K = nviseq.K[0]
    H, W = nviseq.img_h, nviseq.img_w
    
    ## load part data
    parts = []
    obj = ArtiPart(None, None, None, DEVICE, name='obj')
    obj.subparts = []
    for i in range(args.part_cnt):
        part_data = np.load(
            f'{args.fit_ckpt}/part_{i}_partdata.npy', 
            allow_pickle=True
        ).item()
        part = ArtiPart(None, None, None, DEVICE, name=f'part_{i}')
        part.load_dict(part_data)
        obj.subparts.append(part)
        parts.append(part)
    allpart_params = []
    for part_id, part in enumerate(parts):
        param_dump = np.load(
            f'{args.fit_ckpt}/part_{part_id}_best.npy', 
            allow_pickle=True
        )
        part_param = [TransformParams(device=DEVICE, fit_scale=False, train=False) 
                        for _ in range(len(param_dump))]
        for i, p in enumerate(param_dump):
            part_param[i].set_param(p)
        allpart_params.append(part_param)
    loguru.critical(f'Loaded part params from {args.fit_ckpt} checkpoint path')
    
    # load hands
    hparam : dict[str, MANOParams] = {}
    if args.load_mano_raw:
        mano_params = np.load(
            f'{args.seq_path}/processed/hamer/manoparam_fit.slerp.npy', 
            allow_pickle=True
        ).item()
    
        for s, h_param in mano_params.items():
            if h_param is None or not isinstance(h_param, dict):
                continue
            handedness = _handedness(s, h_param)
            ## solve mean pose
            h_param['betas'] = np.mean(h_param["betas"], axis=0)[None, :].repeat(F, axis=0)
            hp = MANOParams(
                h_param,
                hand_type=handedness,
                K=K, w2c=w2c,
                device=DEVICE
            )
            hp.freeze_global()
            hparam[s] = hp
    else:
        hands_dict = np.load(
            f'{args.fit_ckpt}/data_hand.npy', 
            allow_pickle=True
        ).item()
        for s, mano_param_d in hands_dict.items():
            if mano_param_d is None:
                loguru.warning(f'No {s} hand found, skipping...')
                continue
            handedness = _handedness(s, mano_param_d)
            flat_hand_mean = bool(mano_param_d.get('flat_hand_mean', False))
            betas = to_tensor(mano_param_d['mean_shape'], device=DEVICE)
            mano_param_d = {
                "betas": torch.tile(betas, (F, 1)),  # repeat betas for each frame
                "transl": to_tensor(mano_param_d['hand_trans'], device=DEVICE),
                "global_orient": to_tensor(mano_param_d['hand_poses'][:, :3], device=DEVICE),
                "hand_pose": to_tensor(mano_param_d['hand_poses'][:, 3:], device=DEVICE),
                "flat_hand_mean": flat_hand_mean,
            }
            # arrgh(mano_param_d['betas'], mano_param_d['global_orient'], 
            #       mano_param_d['hand_pose'], mano_param_d['transl'])
            # breakpoint()
            mano_param = MANOParams(
                mano_param_d,
                hand_type=handedness,
                K=K, w2c=w2c,
                device=DEVICE,
            )
            mano_param.freeze_global()
            hparam[s] = mano_param
    
    return viseq, nviseq, parts, allpart_params, hparam

def dump_ho_data(
    hp : dict[str, MANOParams], # {'left': MANOParams, 'right': MANOParams}
    parts_tfs : list[list[TransformParams]], # [part_cnt, F] of TransformParams
    parts : list[ArtiPart], # list of ArtiPart
    args,
    out_iter=None, 
):
    if out_iter is not None:
        dpath = f"{args.output_path}/iter_{out_iter}"
    else:
        dpath = args.output_path
    os.makedirs(dpath, exist_ok=True)
    F = len(parts_tfs[0])  # number of frames
    for part_id, part in enumerate(parts):
        params = parts_tfs[part_id]
        param_dump = []
        for i, p in enumerate(params):
            param_dump.append(p.get_param())
        np.save(f"{dpath}/part_{part_id}_best.npy", 
                param_dump, allow_pickle=True
        )
        np.save(f"{dpath}/part_{part_id}_partdata.npy", 
                part.dump_dict(), allow_pickle=True
        )
    import hrse.MANO.generate_hand_temp as generate_hand_temp
    generate_hand_temp.dump_instance_params(hp, dpath, F)
    loguru.critical(f'Dumped params to {dpath}')
