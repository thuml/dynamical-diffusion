import os
import torch
import torchmetrics
import numpy as np

from dydiff.dydiff import DynamicalLDMCleanedWithEncoderCondition, DynamicalLatentDiffusion

from logger.visualization_turbulence import save_plots
from datasets.normalizer import NullNormalizer


class DynamicalLDMForTurbulenceWithEncoderCondition(DynamicalLDMCleanedWithEncoderCondition):
    def __init__(self, *args,
                 total_length=24,
                 input_length=4,
                 num_vis=10,
                 validation_save_dir="",
                 validate_kwargs={},
                 unconditional_guidance_scale=1.,
                 visualize_intermediates=False,
                 rollout=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.input_length = input_length
        self.total_length = total_length
        self.num_vis = num_vis

        self.validation_save_dir = validation_save_dir
        self.validate_kwargs = validate_kwargs
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.visualize_intermediates = visualize_intermediates
        self.rollout = rollout

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        # self.rollout_mse = [torchmetrics.MeanSquaredError() for _ in range(1, self.rollout)]
        # self.rollout_mae = [torchmetrics.MeanAbsoluteError() for _ in range(1, self.rollout)]
    
    # def test_step(self, batch, batch_idx):
    #     self.validation_step(batch, batch_idx)
    
    @torch.no_grad()
    def get_input(self, batch, k, k_prev, log_mode=False):
        
        if self.model.conditioning_key is not None and not self.force_null_conditioning:
            x_c = super(DynamicalLatentDiffusion, self).get_input(batch, self.cond_stage_key).to(self.device)
            z_c = self.batched_encode(x_c)
        else:
            x_c, z_c = None, None
            
        x = super(DynamicalLatentDiffusion, self).get_input(batch, k).to(self.device)
        if self.model.conditioning_key.startswith('concat-video-mask'):
            # print(x.shape, x_c.shape, z_c.shape)
            x = torch.cat([x_c, x], dim=1)
            x_prev = super(DynamicalLatentDiffusion, self).get_input(batch, k_prev).to(self.device)
            if getattr(self, "input_length", None) is None:
                self.input_length = x_c.shape[1] // self.x_channels
            elif self.input_length != x_c.shape[1] // self.x_channels:
                raise ValueError("input_length does not match conditioning shape, got {} and {}".format(self.input_length, x_c.shape[1] // self.z_channels))
        else:
            x_prev = super(DynamicalLatentDiffusion, self).get_input(batch, k_prev).to(self.device)

        z, z_prev = self.batched_encode(x), self.batched_encode(x_prev)

        out = [z, z_prev, z_c]
        
        if log_mode:
            x_rec = self.batched_decode(z)
            x_prev_rec = self.batched_decode(z_prev)
            x_c_rec = self.batched_decode(z_c)
            out.extend([x, x_prev, x_rec, x_prev_rec, x_c, x_c_rec])
        return out
    
    def validation_step(self, batch, batch_idx):
        z, z_prev, z_c, x, x_prev, x_rec, x_prev_rec, x_c, x_c_rec = self.get_input(batch, self.first_stage_key, self.first_stage_key_prev, log_mode=True)
        
        unconditional_guidance_scale = self.unconditional_guidance_scale
        if unconditional_guidance_scale > 1:
            uc = self.get_unconditional_conditioning(batch)
        else:
            uc = None
        
        with self.ema_scope():
            z_sample, (z_intermediates, z_x0s) = self.sample_log(x_prev=z_prev, cond=z_c, batch_size=z.shape[0], 
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=uc,
                                                        **self.validate_kwargs)
        x_sample = self.batched_decode(z_sample)

        self.valid_mse(NullNormalizer.denormalize(x_sample), NullNormalizer.denormalize(x))
        self.valid_mae(NullNormalizer.denormalize(x_sample), NullNormalizer.denormalize(x))

        if batch_idx < self.num_vis:
            self.add_visualization_case(
                # [ground truth, prediction]
                visualization_data=[
                    # NullNormalizer.denormalize(x_prev)[0].cpu().numpy(),
                    # NullNormalizer.denormalize(x_prev_rec)[0].cpu().numpy(),
                    NullNormalizer.denormalize(x)[0].cpu().numpy(),
                    # NullNormalizer.denormalize(x_rec)[0].cpu().numpy(),
                    NullNormalizer.denormalize(x_sample)[0].cpu().numpy(),
                ],
                local_save_root=os.path.join(self.validation_save_dir, 'val/{}/{}-{}'.format(self.global_step, batch_idx, self.local_rank)),
                tensorboard_log_root='val/visualization/{}'.format(batch_idx)
            )

            if self.visualize_intermediates:
                x_intermediates = [self.batched_decode(z_intermediate) for z_intermediate in z_intermediates]
                x_intermediates_visualization = [
                    NullNormalizer.denormalize(x_intermediate)[0].cpu().numpy()
                    for x_intermediate in x_intermediates
                ]

                x_x0s = [self.batched_decode(z_x0) for z_x0 in z_x0s]
                x_x0s_visualization = [
                    NullNormalizer.denormalize(x_x0)[0].cpu().numpy()
                    for x_x0 in x_x0s
                ]

                self.add_visualization_case(
                    visualization_data=x_intermediates_visualization,
                    local_save_root=os.path.join(self.validation_save_dir, 'val/{}/{}-{}/intermediates'.format(self.global_step, batch_idx, self.local_rank)),
                    tensorboard_log_root='val/visualization/{}'.format(batch_idx)
                )

                self.add_visualization_case(
                    visualization_data=x_x0s_visualization,
                    local_save_root=os.path.join(self.validation_save_dir, 'val/{}/{}-{}/x0s'.format(self.global_step, batch_idx, self.local_rank)),
                    tensorboard_log_root='val/visualization/{}'.format(batch_idx)
                )
    
    def test_step(self, batch, batch_idx):
        z, z_prev, z_c, x, x_prev, x_rec, x_prev_rec, x_c, x_c_rec = self.get_input(batch, self.first_stage_key, self.first_stage_key_prev, log_mode=True)
        x_total = super(DynamicalLatentDiffusion, self).get_input(batch, 'total').to(self.device)
        z_total = self.batched_encode(x_total)
        
        unconditional_guidance_scale = self.unconditional_guidance_scale
        if unconditional_guidance_scale > 1:
            uc = self.get_unconditional_conditioning(batch)
        else:
            uc = None
        
        with self.ema_scope():
            z_sample, (z_intermediates, z_x0s) = self.sample_log(x_prev=z_prev, cond=z_c, batch_size=z.shape[0], 
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=uc,
                                                        **self.validate_kwargs)
        x_sample = self.batched_decode(z_sample)
        # print(x.shape, x_sample.shape, x_prev.shape, x_c.shape, x_total.shape)
        # assert False

        self.valid_mse(NullNormalizer.denormalize(x_sample), NullNormalizer.denormalize(x))
        self.valid_mae(NullNormalizer.denormalize(x_sample), NullNormalizer.denormalize(x))
        
        # ==============================================================================================================
        # save gt and pred (100 samples) for evaluation
        save_root = os.path.join(self.validation_save_dir, f'output_for_evaluation_{self.global_step}')
        os.makedirs(save_root, exist_ok=True)
        x_gather = self.concat_all_gather(x)
        if self.local_rank == 0:
            x_gather = NullNormalizer.denormalize(x_gather)
            x_gather = x_gather.view(-1, self.total_length, self.x_channels, 64, 64).cpu().numpy()
            np.save(os.path.join(save_root, 'gt_{}.npy'.format(batch_idx)), x_gather)

        print(batch_idx)
        for sample_idx in range(8):
            with self.ema_scope():
                z_sample_i, _ = self.sample_log(x_prev=z_prev, cond=z_c, batch_size=z.shape[0],
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=uc,
                                                **self.validate_kwargs)
            x_sample_i = self.batched_decode(z_sample_i)
            x_sample_i_gather = self.concat_all_gather(x_sample_i)
            if self.local_rank == 0:
                x_sample_i_gather = NullNormalizer.denormalize(x_sample_i_gather)
                x_sample_i_gather = x_sample_i_gather.view(-1, self.total_length, self.x_channels, 64, 64).cpu().numpy()
                np.save(os.path.join(save_root, 'pred_{}_sample_{}.npy'.format(batch_idx, sample_idx)),
                        x_sample_i_gather)
        # ==============================================================================================================

        # if self.rollout > 1:
        #     sample_channels = (self.total_length - self.input_length) * self.z_channels
        #     for rollout_t in range(1, self.rollout):
        #         z_c = torch.cat([z_c[:, sample_channels:], z_sample], dim=1)
        #         with self.ema_scope():
        #             z_sample, _ = self.sample_log(x_prev=z_sample, cond=z_c, batch_size=z.shape[0], 
        #                                           unconditional_guidance_scale=unconditional_guidance_scale,
        #                                           unconditional_conditioning=uc,
        #                                           **self.validate_kwargs)
        #         x_sample_i = self.batched_decode(z_sample)
        #         x_sample = torch.cat([x_sample, x_sample_i], dim=1)
        #         self.rollout_mse[rollout_t-1](NullNormalizer.denormalize(x_sample_i).cpu(), 
        #                         NullNormalizer.denormalize(x_total[:,(self.input_length+rollout_t) * self.x_channels:(self.input_length+rollout_t+1) * self.x_channels]).cpu())
        #         self.rollout_mae[rollout_t-1](NullNormalizer.denormalize(x_sample_i).cpu(), 
        #                         NullNormalizer.denormalize(x_total[:,(self.input_length+rollout_t) * self.x_channels:(self.input_length+rollout_t+1) * self.x_channels]).cpu())
        
        # if self.total_length > 11:
        # x_sample = x_sample.view(x.shape[0], -1, self.x_channels, 64, 64)
        # self.add_visualization_case(
        #     # [ground truth, prediction]
        #     visualization_data=NullNormalizer.denormalize(x_sample)[0].cpu().numpy(),
        #     local_save_root=os.path.join(self.validation_save_dir, 'test/{}/{}-{}'.format(self.global_step, batch_idx, self.local_rank)),
        #     tensorboard_log_root='test/visualization/{}'.format(batch_idx)
        # )
            
        # if batch_idx < self.num_vis:
        #     self.add_visualization_case(
        #         # [ground truth, prediction]
        #         visualization_data=[
        #             # NullNormalizer.denormalize(x_prev)[0].cpu().numpy(),
        #             # NullNormalizer.denormalize(x_prev_rec)[0].cpu().numpy(),
        #             NullNormalizer.denormalize(x)[0].cpu().numpy(),
        #             # NullNormalizer.denormalize(x_rec)[0].cpu().numpy(),
        #             NullNormalizer.denormalize(x_sample)[0,:self.x_channels].cpu().numpy(),
        #         ],
        #         local_save_root=os.path.join(self.validation_save_dir, 'test/{}/{}-{}'.format(self.global_step, batch_idx, self.local_rank)),
        #         tensorboard_log_root='test/visualization/{}'.format(batch_idx)
        #     )
            
        #     # if self.rollout > 1:
        #     #     x_sample = x_sample.view(-1, self.rollout, self.x_channels, 64, 64)
        #     #     self.add_visualization_case(
        #     #         visualization_data=NullNormalizer.denormalize(x_sample)[0].cpu().numpy(),
        #     #         local_save_root=os.path.join(self.validation_save_dir, 'test/{}/{}-{}/rollout'.format(self.global_step, batch_idx, self.local_rank)),
        #     #         tensorboard_log_root='test/visualization/{}'.format(batch_idx)
        #     #     )
        #     #     x_rollout_gt = x_total[:,self.input_length * self.x_channels:min(self.input_length+self.rollout, 20) * self.x_channels].view(-1, self.rollout, self.x_channels, 64, 64)
        #     #     self.add_visualization_case(
        #     #         visualization_data=NullNormalizer.denormalize(x_rollout_gt)[0].cpu().numpy(),
        #     #         local_save_root=os.path.join(self.validation_save_dir, 'test/{}/{}-{}/rollout_gt'.format(self.global_step, batch_idx, self.local_rank)),
        #     #         tensorboard_log_root='test/visualization/{}'.format(batch_idx)
        #     #     )

        #     if self.visualize_intermediates and not self.rollout:
        #         x_intermediates = [self.batched_decode(z_intermediate) for z_intermediate in z_intermediates]
        #         x_intermediates_visualization = [
        #             NullNormalizer.denormalize(x_intermediate)[0].cpu().numpy()
        #             for x_intermediate in x_intermediates
        #         ]

        #         x_x0s = [self.batched_decode(z_x0) for z_x0 in z_x0s]
        #         x_x0s_visualization = [
        #             NullNormalizer.denormalize(x_x0)[0].cpu().numpy()
        #             for x_x0 in x_x0s
        #         ]

        #         self.add_visualization_case(
        #             visualization_data=x_intermediates_visualization,
        #             local_save_root=os.path.join(self.validation_save_dir, 'test/{}/{}-{}/intermediates'.format(self.global_step, batch_idx, self.local_rank)),
        #             tensorboard_log_root='test/visualization/{}'.format(batch_idx)
        #         )

        #         self.add_visualization_case(
        #             visualization_data=x_x0s_visualization,
        #             local_save_root=os.path.join(self.validation_save_dir, 'test/{}/{}-{}/x0s'.format(self.global_step, batch_idx, self.local_rank)),
        #             tensorboard_log_root='test/visualization/{}'.format(batch_idx)
        #         )
    
    def on_validation_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()

        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def on_test_epoch_end(self):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()

        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
        
        if self.rollout:
            for i in range(1, self.rollout):
                mse = self.rollout_mse[i-1].compute()
                mae = self.rollout_mae[i-1].compute()
                self.log('valid_mse_rollout_{}'.format(i+1), mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.log('valid_mae_rollout_{}'.format(i+1), mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
                self.rollout_mse[i-1].reset()
                self.rollout_mae[i-1].reset()

    def concat_all_gather(self, x):
        x = self.all_gather(x)
        if isinstance(x, torch.Tensor):
            return x
        x = torch.concat([_.to(self.device) for _ in x])
        return x
    
    def add_visualization_case(self, visualization_data, local_save_root, tensorboard_log_root):
        tb_logger = self.trainer.logger.experiment
        if tb_logger is None:
            raise ValueError('TensorBoard logger not found')

        save_plots(
            fig_names=['mmhr_pred_{:0>2d}.png'.format(i + 1) for i in
                       range(self.total_length)],
            outputs=np.concatenate(visualization_data, axis=2),
            save_root=local_save_root,
        )
