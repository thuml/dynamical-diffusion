"""
Modified from https://github.com/thuml/iVideoGPT/blob/e9eaeee33213e41d60927c6574812ac67c0e432f/ivideogpt/utils/video_metric.py
"""

import argparse
import os
from typing import Tuple

import tqdm
from tqdm import trange
import scipy
import scipy.linalg
import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from metrics_vid_pred import Metrics

# import piqa
# import lpips


def batch_forward(batch_size, input1, input2, forward, verbose=False):
    assert input1.shape[0] == input2.shape[0]
    return torch.cat([forward(input1[i: i + batch_size], input2[i: i + batch_size]) for i in
                      trange(0, input1.shape[0], batch_size, disable=not verbose)], dim=0)


def batch_forward_for_fvd(batch_size, input, forward, verbose=False):
    return torch.cat(
        [forward(input[i: i + batch_size]) for i in trange(0, input.shape[0], batch_size, disable=not verbose)], dim=0)


class Evaluator(nn.Module):
    def __init__(self, i3d_path=None, detector_kwargs=None, max_batchsize=None):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lpips = lpips.LPIPS(net='vgg')
        self.psnr = piqa.PSNR(epsilon=1e-08, value_range=1.0, reduction='none')
        self.ssim = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction='none')

        self.i3d_model = torch.jit.load(i3d_path).eval()
        self.max_batchsize = max_batchsize

    def compute_fvd(self, real_feature, gen_feature):
        if real_feature.num_items == 0 or gen_feature.num_items == 0:
            raise ValueError("No data to compute FVD")

        mu_real, sigma_real = real_feature.get_mean_cov()
        mu_gen, sigma_gen = gen_feature.get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    def compute_fvd_from_raw_data(self, real_data=None, gen_data=None):

        detector_kwargs = dict(rescale=True, resize=True,
                               return_features=True)  # Return raw features before the softmax layer.

        mu_real, sigma_real = compute_feature_stats_for_dataset(self.i3d_model, detector_kwargs=detector_kwargs,
                                                                data=real_data).get_mean_cov()

        mu_gen, sigma_gen = compute_feature_stats_for_dataset(self.i3d_model, detector_kwargs=detector_kwargs,
                                                              data=gen_data).get_mean_cov()

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    def forward(self, video_1, video_2):
        # video_1: ground-truth
        # video_2: reconstruction or prediction

        if video_1.shape[0] < video_2.shape[0]:
            B, T, C, H, W = video_1.shape
            t = video_2.shape[0] // B
            video_1 = video_1.repeat([t, 1, 1, 1, 1])

            video_1 = video_1.reshape(-1, C, H, W)
            video_2 = video_2.reshape(-1, C, H, W)

            mse = self.mse(video_1, video_2).mean([1, 2, 3])
            psnr = self.psnr(video_1, video_2)
            if self.max_batchsize is not None and video_1.shape[0] > self.max_batchsize:
                ssim = batch_forward(
                    self.max_batchsize,
                    video_1, video_2,
                    lambda x1, x2: self.ssim(x1, x2)
                )
            else:
                ssim = self.ssim(video_1, video_2)
            if self.max_batchsize is not None and video_1.shape[0] > self.max_batchsize:
                lpips = batch_forward(
                    self.max_batchsize,
                    video_1 * 2 - 1, video_2 * 2 - 1,
                    lambda x1, x2: self.lpips(x1, x2).mean((1, 2, 3)),
                )
            else:
                lpips = self.lpips(video_1 * 2 - 1, video_2 * 2 - 1).mean((1, 2, 3))

            # get best of t predictions
            return (
                mse.reshape(t, B, T).mean(-1).min(0).values.mean(),
                psnr.reshape(t, B, T).mean(-1).max(0).values.mean(),
                ssim.reshape(t, B, T).mean(-1).max(0).values.mean(),
                lpips.reshape(t, B, T).mean(-1).min(0).values.mean(),
            )
        else:
            B, T, C, H, W = video_1.shape
            video_1 = video_1.reshape(B * T, C, H, W)
            video_2 = video_2.reshape(B * T, C, H, W)

            return (
                self.mse(video_1, video_2).mean(),
                self.psnr(video_1, video_2).mean(),
                self.ssim(video_1, video_2).mean(),
                self.lpips(video_1 * 2 - 1, video_2 * 2 - 1).mean(),
            )


@torch.no_grad()
def compute_feature_stats_for_dataset(detector, detector_kwargs, data=None):
    stats = FeatureStats(capture_mean_cov=True)

    for i in range(data.size(0)):
        # [batch_size, c, t, h, w]
        images = data[i].permute(0, 2, 1, 3, 4).contiguous()
        if images.shape[1] == 1:
            images = images.repeat([1, 3, *([1] * (images.ndim - 2))])
        features = detector(images, **detector_kwargs)
        stats.append_torch(features)

    return stats


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x):
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


def create_test_dataset(dataset_root, max_items=None, num_samples=None):
    class VideoDataset(Dataset):
        def __init__(self, root, max_items=None, num_samples=100):
            """
            Save format
            - gt_{idx}.npy
            - pred_{idx}_sample_{sample_idx}.npy (sample_idx in [0, 99])
            """
            self.root = root
            save_file_list = os.listdir(root)
            self.n_test_cases = int(sum(['gt' in file for file in save_file_list]))
            self.max_items = max_items if max_items is not None else self.n_test_cases
            self.num_samples = num_samples

        def __len__(self):
            return min(self.n_test_cases, self.max_items)

        def __getitem__(self, idx):
            gt = np.load(os.path.join(self.root, f'gt_{idx}.npy'))
            pred = []
            for sample_idx in range(self.num_samples):
                pred.append(np.load(os.path.join(self.root, f'pred_{idx}_sample_{sample_idx}.npy')))
            pred = np.concatenate(pred, axis=0)
            return gt, pred

    return VideoDataset(dataset_root, max_items, num_samples)


def evaluate(args):
    # evaluator = Evaluator(args.i3d_model_path, max_batchsize=args.max_batchsize).cuda()
    dataset = create_test_dataset(os.path.join(args.model_output_root, args.model_output_dir),
                                  max_items=args.max_items, num_samples=args.num_samples)
    test_data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    mse_values, psnr_values, ssim_values, lpips_values = [], [], [], []
    gt_features, pred_features = FeatureStats(capture_mean_cov=True), FeatureStats(capture_mean_cov=True)
    
    metric = None
    with torch.no_grad():
        for gt, pred in tqdm.tqdm(test_data_loader):
            gt = gt.squeeze().cuda()
            pred = pred.squeeze().cuda()
            gt = gt[:, 4:]
            pred = pred[:, 4:]
            
            
            B, T, C, H, W = gt.shape
            t = pred.shape[0] // B
            gt_all = gt.repeat([t, 1, 1, 1, 1])
            
            if metric is None:
                metric = Metrics(H, W, T*C, 'cuda', [0.0], t)
            metric.add_test_case(pred.reshape(pred.shape[0], T*C, H, W), gt_all.reshape(gt_all.shape[0], T*C, H, W))
            # metric.add_test_case(pred, gt_all)
            

            # # compute mse, psnr, ssim, lpips
            # mse, psnr, ssim, lpips = evaluator(gt, pred)
            # mse_values.append(mse.cpu().numpy())
            # psnr_values.append(psnr.cpu().numpy())
            # ssim_values.append(ssim.cpu().numpy())
            # lpips_values.append(lpips.cpu().numpy())


            # # compute fvd
            # detector_kwargs = dict(rescale=True, resize=True, return_features=True)
            # reshape_gt = gt.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, c, t, h, w]
            # if reshape_gt.shape[0] > args.max_batchsize:
            #     gt_features_batch = batch_forward_for_fvd(
            #         args.max_batchsize,
            #         reshape_gt * 255.,
            #         lambda x: evaluator.i3d_model(x, **detector_kwargs)
            #     )
            # else:
            #     gt_features_batch = evaluator.i3d_model(reshape_gt * 255., **detector_kwargs)
            # gt_features.append_torch(gt_features_batch)

            # reshape_pred = pred.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size * 100, c, t, h, w]
            # if reshape_pred.shape[0] > args.max_batchsize:
            #     pred_features_batch = batch_forward_for_fvd(
            #         args.max_batchsize,
            #         reshape_pred * 255.,
            #         lambda x: evaluator.i3d_model(x, **detector_kwargs)
            #     )
            # else:
            #     pred_features_batch = evaluator.i3d_model(reshape_pred * 255., **detector_kwargs)
            # pred_features.append_torch(pred_features_batch)

    # test_mse = np.array(mse_values).mean()
    # test_psnr = np.array(psnr_values).mean()
    # test_ssim = np.array(ssim_values).mean()
    # test_lpips = np.array(lpips_values).mean()
    # print('MSE:', test_mse)
    # print('PSNR:', test_psnr)
    # print('SSIM:', test_ssim * 100)
    # print('LPIPS:', test_lpips * 100)

    # fvd = evaluator.compute_fvd(gt_features, pred_features)
    # print("FVD:", fvd)
    
    results = metric.cal()
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output_root', type=str)
    parser.add_argument('--model_output_dir', type=str, default='')
    parser.add_argument('--i3d_model_path', type=str)
    parser.add_argument('--max_batchsize', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--max_items', type=int, default=50)
    args = parser.parse_args()
    evaluate(args)
