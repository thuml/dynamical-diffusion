from abc import abstractmethod
from typing import Optional
import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.video_attention import SpatialVideoTransformer
from ldm.util import exists

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from einops import rearrange


def make_grid(input):
    B, C, H, W = input.size()
    xx = th.arange(0, W, device=input.device).view(1, -1).repeat(H, 1)
    yy = th.arange(0, H, device=input.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = th.cat((xx, yy), 1).float()
    return grid

def warp(input, flow, grid, mode="bilinear", padding_mode="border"):
    B, C, H, W = input.size()
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    # output = torch.nn.functional.grid_sample(input, vgrid)
    output = F.grid_sample(input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
    return output

class WarpingUNetModel(UNetModel):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,  # duplicated in ControlNet
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        x_channels=3,
    ):
        self.motion_channels = out_channels // x_channels * 2
        self.density_channels = out_channels
        self.x_channels = x_channels
        self.num_frames = out_channels // x_channels
        super().__init__(
            image_size,
            in_channels,
            model_channels,
            self.motion_channels + self.density_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
            use_spatial_transformer,
            transformer_depth,
            context_dim,
            n_embed,
            legacy,
            disable_self_attentions,
            num_attention_blocks,
            disable_middle_self_attn,
            use_linear_in_transformer,
        )
    
    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        out = super().forward(x, timesteps, context, y, **kwargs)
        # print(out.shape, self.motion_channels)
        motion, residual = out[:, :self.motion_channels], out[:, self.motion_channels:]
        last_frame = x[:, -self.x_channels:]
        grid = make_grid(last_frame)
        output_list = []
        for i in range(self.num_frames):
            motion_i = motion[:, i * 2: (i + 1) * 2]
            residual_i = residual[:, i * self.x_channels: (i + 1) * self.x_channels]
            # print(last_frame.shape, motion_i.shape, residual_i.shape, grid.shape)
            last_frame = warp(last_frame, motion_i, grid) + residual_i
            output_list.append(last_frame)
        output = th.cat(output_list, 1)
        return output