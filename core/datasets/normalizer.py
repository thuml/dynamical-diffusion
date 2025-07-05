import torch
import numpy as np


class NullNormalizer:
    """
    Identity normalizer.
    """
    @staticmethod
    def normalize(x: torch.Tensor):
        return x
    
    @staticmethod
    def denormalize(v: torch.Tensor):
        return v


class PrecipitationNormalizer:
    """
    First transform input x into log_10(1+x), then normalize into [vmin, vmax].
    """
    xmin = 0.
    xmax = 128.
    log_xmin = np.log10(1 + xmin)
    log_xmax = np.log10(1 + xmax)
    vmin = -1.
    vmax = 1.
    
    @classmethod
    def normalize(cls, x: torch.Tensor, affine: bool = True):
        x = torch.clamp(x, cls.xmin, cls.xmax)
        log_x = torch.log10(x + 1)
        if affine:
            return (log_x - cls.log_xmin) / (cls.log_xmax - cls.log_xmin) * (cls.vmax - cls.vmin) + cls.vmin
        else:
            return log_x
    
    @classmethod
    def denormalize(cls, v: torch.Tensor, affine: bool = True):
        v = torch.clamp(v, cls.vmin, cls.vmax)
        if affine:
            log_x = (v - cls.vmin) / (cls.vmax - cls.vmin) * (cls.log_xmax - cls.log_xmin) + cls.log_xmin
        else:
            log_x = v
        return 10 ** log_x - 1


class RGBNormalizer:
    """
    Normalize RGB images from [0, 1] to [-1, 1].
    """
    x_min = 0.
    x_max = 1.
    v_min = -1.
    v_max = 1.

    @staticmethod
    def normalize(x: torch.Tensor):
        x = torch.clamp(x, RGBNormalizer.x_min, RGBNormalizer.x_max)
        return x * 2 - 1
    
    @staticmethod
    def denormalize(v: torch.Tensor):
        v = torch.clamp(v, RGBNormalizer.v_min, RGBNormalizer.v_max)
        return (v + 1) / 2
