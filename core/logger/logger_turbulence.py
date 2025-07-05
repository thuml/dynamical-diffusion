import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from datasets.normalizer import NullNormalizer
from logger.visualization_turbulence import save_plots


def save_img(visualization_data, path, filename_prefix):
    # visualization_data: (c, h, W)
    filenames = [f"{filename_prefix}_{i:0>2d}.png" for i in range(visualization_data.shape[0])]
    save_plots(filenames, visualization_data, path)


class ImageLoggerWithKeyToConcat(Callback):
    def __init__(self, save_dir='results', keys_to_concat=None, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.save_dir = save_dir
        self.keys_to_concat = keys_to_concat
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(self.save_dir, "image_log", split, "step-{:06}".format(global_step))
        if self.keys_to_concat is not None:
            data_to_plot = torch.cat([images[k] for k in self.keys_to_concat], dim=3)
            data_to_plot = NullNormalizer.denormalize(data_to_plot)
            for idx, datum_to_plot in enumerate(data_to_plot):
                _, h, w = datum_to_plot.shape
                datum_to_plot = datum_to_plot.reshape(-1, 2, h, w)
                sample_idx = batch_idx * data_to_plot.shape[0] + idx
                batch_root = os.path.join(root, str(sample_idx))
                filename = "{:06}".format(sample_idx)
                save_img(datum_to_plot.cpu().numpy(), batch_root, filename)

        for k in images:
            if k != "diffusion_row":
                data_to_plot = images[k]
                data_to_plot = NullNormalizer.denormalize(data_to_plot)
                for idx, datum_to_plot in enumerate(data_to_plot):
                    sample_idx = batch_idx * data_to_plot.shape[0] + idx
                    batch_root = os.path.join(root, k, str(sample_idx))
                    filename = "{:06}".format(sample_idx)
                    save_img(datum_to_plot.cpu().numpy(), batch_root, filename)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, plot_progressive_rows=False, **self.log_images_kwargs)
                images = {k:v for k, v in images.items() if not k.startswith("z_")}

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(self.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
