import os
import pytorch_lightning as pl
from dydiff.datasets.turbulence import TurbulenceDataModuleForDyDiff

from logger.logger_turbulence import ImageLoggerWithKeyToConcat

import argparse
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from pytorch_lightning.callbacks import ModelCheckpoint


def create_data_module(cfg):
    data_module = TurbulenceDataModuleForDyDiff(cfg.data, cfg.training.batch_size, cfg.training.num_workers)
    return data_module

def create_model(cfg):
    cfg.model.params.total_length = cfg.data.total_length
    cfg.model.params.input_length = cfg.data.input_length
    cfg.model.params.num_vis = cfg.eval.num_vis
    cfg.model.params.validation_save_dir = cfg.training.logger.save_dir

    cfg.model.params.channels = cfg.data.total_length - cfg.data.input_length
    cfg.model.params.unet_config.params.in_channels += cfg.data.total_length
    cfg.model.params.unet_config.params.out_channels = cfg.data.total_length - cfg.data.input_length
    
    if "ensemble" in cfg.model.params.keys() and cfg.model.params.ensemble == True:
        # cfg.model.params.channels *= 2
        cfg.model.params.unet_config.params.in_channels += cfg.model.params.unet_config.params.out_channels

    cfg.model.params.channels *= 3
    cfg.model.params.unet_config.params.in_channels *= 3
    cfg.model.params.unet_config.params.out_channels *= 3
    if cfg.model.params.conditioning_key == "mcvd":
        cfg.model.params.unet_config.params.cond_channels *= 3
        
    if cfg.model.params.conditioning_key == "concat-video":
        cfg.model.params.unet_config.params.in_channels = (cfg.data.input_length + 1) * 3
        cfg.model.params.unet_config.params.out_channels = 3
        cfg.model.params.unet_config.params.num_video_frames = cfg.data.total_length - cfg.data.input_length
    elif cfg.model.params.conditioning_key.startswith("concat-video-mask"):
        cfg.model.params.unet_config.params.in_channels = 3 * 2 + 1 # 3 * 2 for video&concat, 1 for mask
        if "1st" in cfg.model.params.conditioning_key:
            cfg.model.params.unet_config.params.in_channels += 1
        cfg.model.params.unet_config.params.out_channels = 3
        cfg.model.params.unet_config.params.num_video_frames = cfg.data.total_length
    
    cfg.model.params.ckpt_path = cfg.ckpt_path

    model = instantiate_from_config(cfg.model)
    for k, v in cfg.training.model_attrs.items():
        setattr(model, k, v)
    return model

def create_loggers(cfg):
    image_logger = ImageLoggerWithKeyToConcat(
        batch_frequency=cfg.training.logger.logger_freq,
        save_dir=cfg.training.logger.save_dir,
        keys_to_concat=["inputs", "samples"],
        log_images_kwargs=dict(cfg.model.params.validate_kwargs)
    )
    checkpoint_logger = ModelCheckpoint(
        dirpath=cfg.training.logger.save_dir,
        every_n_train_steps=cfg.training.logger.checkpoint_freq,
        save_top_k=-1
    )
    return [image_logger, checkpoint_logger]


if __name__=='__main__':
    # Parse arguments
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--model_root', type=str)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument("--test", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(cfg, vars(args))
    config_file_name = os.path.basename(args.config_file)
    config_file_name = config_file_name.split('.')[0]

    cfg.training.logger.save_dir = os.path.join(cfg.training.logger.save_dir, config_file_name)
    
    print(OmegaConf.to_yaml(cfg))
    # model
    model = create_model(cfg)

    # data module
    data_module = create_data_module(cfg)

    # loggers
    loggers = create_loggers(cfg)

    # trainer
    trainer = pl.Trainer(gpus=args.n_gpu, precision=32,
                         callbacks=loggers,
                         default_root_dir=cfg.training.logger.save_dir,
                         max_steps=cfg.training.max_iterations,
                         accumulate_grad_batches=cfg.training.accumulate_grad_batches,
                         val_check_interval=int(cfg.training.validation_freq) if cfg.training.validation_freq is not None else float(1.),
                         strategy="ddp")
    # Train!
    if not args.test:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    else:
        trainer.test(model, datamodule=data_module, ckpt_path=args.resume)
