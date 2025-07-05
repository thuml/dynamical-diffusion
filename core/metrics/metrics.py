import torch
import torch.nn.functional as F

import numpy as np
from functools import partial

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import os

def save_metrics(metrics, save_path, label='gan'):
    os.makedirs(save_path, exist_ok=True)
    wave_length = max(metrics['height'], metrics['width'])
    with open(os.path.join(save_path, "metrics.txt"), "w") as f:
        for key, value in metrics.items():
            if 'crps' in key:
                f.write(f"{key}: {value:.6f}\n")
            elif 'rank_histogram' in key:
                bin_prob_list, bin_cum_prob_list = value
                # draw rank histogram with matplotlib
                plt.figure(figsize=(8, 8))
                plt.subplot(2, 1, 1)
                plt.plot(range(len(bin_prob_list)), bin_prob_list, label=label)
                plt.plot(range(len(bin_cum_prob_list)), [0.01] * len(bin_cum_prob_list), label="Ideal", color='gray', linestyle='dashed')
                plt.xlabel("Normalized Rank")
                plt.ylabel("Probability")
                plt.legend()
                plt.subplot(2, 1, 2)
                plt.plot(range(len(bin_cum_prob_list)), bin_cum_prob_list, label=label)
                plt.plot(range(len(bin_cum_prob_list)), [0.01 * i for i in range(len(bin_cum_prob_list))], label="Ideal", color='gray', linestyle='dashed')
                plt.xlabel("Normalized Rank")
                plt.ylabel("Cumulative Probability")
                plt.legend()
                plt.savefig(os.path.join(save_path, "rank_histogram.png"))
                plt.close()
            elif 'ralsd' in key:
                PSD_pd_avg, PSD_gt_avg, lsd = value
                plt.figure(figsize=(8, 6))
                # plt.plot(range(len(PSD_pd_avg)), PSD_pd_avg, label=label)
                # plt.plot(range(len(PSD_gt_avg)), PSD_gt_avg, label="ground truth")
                x = wave_length / np.arange(1, len(PSD_pd_avg))
                log_x = -np.log10(x)
                plt.plot(log_x, PSD_pd_avg[1:], label=label)
                plt.plot(log_x, PSD_gt_avg[1:], label="ground truth")
                plt.xlabel("Wavelength (km)")
                plt.ylabel("PSD (log10)")
                x_label = [2 ** i for i in range(int(np.log2(wave_length) + 1))]
                if wave_length not in x_label:
                    x_label.append(wave_length)
                log_x_label = -np.log10(x_label)
                plt.xticks(log_x_label, x_label)
                plt.legend()
                plt.savefig(os.path.join(save_path, "ralsd.png"))
                plt.close()
                f.write(f"ralsd: {lsd:.6f}\n")
            elif 'rocauc' in key:
                predicts_list, targets_list, probs_list, auc = value
                f.write(f"{key}: {auc:.6f}\n")
            elif 'fss' in key:
                fss, p = value
                f.write(f"{key}: {fss:.6f}, {p:.6f}\n")
            elif 'maemse' in key:
                mae, mse = value
                f.write(f"{key}: {mae:.6f}, {mse:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
                # print('unhandled key:', key)


class CRPS(object):

    def __init__(self, ensemble_num, size=1, type="avg", stride=1):
        self.ensemble_num = ensemble_num
        self.pooling_fn = partial(F.avg_pool2d, kernel_size=size, stride=stride) if type == "avg" else \
                          partial(F.max_pool2d, kernel_size=size, stride=stride)
        self.crps_sum = 0
        self.prob_sum = 0

    def add_test_case(self, pd, gt, probs=1):
        '''
        :param pd: torch.Tensor, size=(N * B, L, H, W)
        :param gt: torch.Tensor, size=(N * B, L, H, W)
        :return: None
        '''
        probs = 1 / probs

        pd, gt = self.pooling_fn(pd), self.pooling_fn(gt)

        _, L, H, W = pd.shape

        pd_ = pd.view(self.ensemble_num, -1, L, H, W)
        gt_ = gt.view(self.ensemble_num, -1, L, H, W)
        
        # l1_loss = torch.sum((torch.abs(pd_ - gt_) * mask), dim=(0,3,4)) / \
        #     torch.sum(mask , dim=(0,2,3,4))

        l1_loss = torch.mean(torch.abs(pd_ - gt_), dim=0)  # (B, L, H, W)

        # print(temp_l1_loss.shape, torch.sum((torch.abs(gt_[0] - pd_[1]) * mask[0]), dim=(1, 2, 3)).shape)
        if self.ensemble_num > 1:
            beta0 = torch.mean(pd_, dim=0)
            pd_sorted, _ = torch.sort(pd_, dim=0, descending=False)
            weight = torch.arange(0, self.ensemble_num).view(-1, 1, 1, 1, 1).to(pd_sorted.device)
            beta1 = torch.sum(weight * pd_sorted, dim=0) / (self.ensemble_num * (self.ensemble_num - 1))
            crps_pixel = l1_loss + beta0 - 2 * beta1
        else:
            crps_pixel = l1_loss
        # print(crps_pixel.shape, mask.shape)

        crps = torch.mean(crps_pixel, dim=(1, 2, 3))
        self.crps_sum += torch.sum(crps * probs).item()
        self.prob_sum += torch.sum(probs).item()

    def calc_result(self):
        return self.crps_sum / self.prob_sum


class RankHistogram(object):

    def __init__(self, ensemble_num, noise=0):
        self.ensemble_num = ensemble_num
        self.noise = noise
        self.normalized_rank_list = []
        self.probs_list = []

    def add_test_case(self, pd, gt, probs=1):
        '''
        :param pd: torch.Tensor, size=(N * B, L, H, W)
        :param gt: torch.Tensor, size=(N * B, L, H, W)
        :return: None
        '''
        probs = 1 / probs

        _, L, H, W = pd.shape

        pd_ = pd.view(self.ensemble_num, -1, L, H, W)
        gt_ = gt.view(self.ensemble_num, -1, L, H, W)

        # add Gaussian noise
        pd_noisy = pd_ + torch.randn_like(pd_) * self.noise
        # gt: mean -> add noise -> repeat -> reshape
        gt_noisy = gt_.mean(dim=0) + torch.randn_like(gt_.mean(dim=0)) * self.noise
        gt_noisy = gt_noisy.repeat(self.ensemble_num, 1, 1, 1).view(self.ensemble_num, -1, L, H, W)

        # calculate normalized rank
        below_gt_mask = (pd_noisy <= gt_noisy).to(torch.float32)  # (N, B, L, H, W)
        normalized_rank = below_gt_mask.sum(dim=0) / self.ensemble_num  # (B, L, H, W)

        # extend probs and get valid probs
        probs = probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, L, H, W)  # (B, L, H, W)

        self.normalized_rank_list.append(normalized_rank.flatten().cpu())
        self.probs_list.append(probs.flatten().cpu())

    def calc_result(self):
        normalized_rank_list = torch.cat(self.normalized_rank_list, dim=0)
        probs_list = torch.cat(self.probs_list, dim=0)
        probs_list /= torch.sum(probs_list)
        # split normalized rank into 100 (actually 101) bins
        normalized_rank_list = normalized_rank_list * 100
        normalized_rank_list = normalized_rank_list.to(torch.int32)
        # calculate bin probability
        bin_prob_list = torch.zeros(101)
        for i in range(101):
            bin_prob_list[i] = torch.sum(probs_list[normalized_rank_list == i])
        # calculate bin cumulative probability
        bin_cum_prob_list = torch.zeros(101)
        bin_cum_prob_list[0] = bin_prob_list[0]
        for i in range(1, 101):
            bin_cum_prob_list[i] = bin_cum_prob_list[i - 1] + bin_prob_list[i]
        # return normalized occurance and cumulative probability
        return bin_prob_list, bin_cum_prob_list

class MAEMSE(object):
    
    def __init__(self, ensemble_num, patch_size=1):
        self.ensemble_num = ensemble_num
        self.patch_size = patch_size
        self.mae_sum = 0
        self.mse_sum = 0
        self.prob_sum = 0

    def add_test_case(self, pd, gt, probs=1):
        '''
        :param pd: torch.Tensor, size=(N * B, L, H, W)
        :param gt: torch.Tensor, size=(N * B, L, H, W)
        :return: None
        '''
        probs = 1 / probs

        _, L, H, W = pd.shape

        pd_ = pd.view(self.ensemble_num, -1, L, H, W)
        gt_ = gt.view(self.ensemble_num, -1, L, H, W)

        # Average over ensemble dimension
        pd_ = torch.mean(pd_, dim=0)  # (B, L, H, W)
        gt_ = torch.mean(gt_, dim=0)  # (B, L, H, W)

        # Average over patch
        pd_ = F.avg_pool2d(pd_, kernel_size=self.patch_size, stride=self.patch_size)  # (B, L, H, W)
        gt_ = F.avg_pool2d(gt_, kernel_size=self.patch_size, stride=self.patch_size)  # (B, L, H, W)

        # Calculate MAE and MSE
        mae = torch.mean(torch.abs(pd_ - gt_), dim=(1, 2, 3))  # (B,)
        mse = torch.mean((pd_ - gt_) ** 2, dim=(1, 2, 3))  # (B,)
        self.mae_sum += torch.sum(mae * probs).item()
        self.mse_sum += torch.sum(mse * probs).item()
        self.prob_sum += torch.sum(probs).item()

    def calc_result(self):
        return self.mae_sum / self.prob_sum, self.mse_sum / self.prob_sum

class RAPSD(object):

    def __init__(self, ensemble_num, height=256, width=256, **fft_kwargs):
        self.ensemble_num = ensemble_num
        YC, XC = compute_centred_coord_array(height, width)
        self.R = torch.tensor(np.sqrt(XC * XC + YC * YC).round()).cuda()
        self.L = max(height, width)
        self.r_range = np.arange(0, (self.L + 1) // 2)

        self.PSD_list = []
        self.probs_list = []

    def add_test_case(self, field, mask, probs=1):
        """
        param field: torch.Tensor, size=(N * B, L, H, W)
        mask: B
        return: None
        """
        probs = 1 / probs
        _, L, H, W = field.shape

        field_ = field.view(self.ensemble_num, -1, L, H, W)

        # Average over ensemble dimension
        field_ = torch.mean(field_, dim=0)  # (B, L, H, W)
        field_ = field_[~mask]  # (B', L, H, W)
        probs = probs[~mask]  # (B',)

        F = torch.fft.fftshift(torch.fft.fft2(field_), dim=(-1, -2))  # (B, L, H, W)
        F = torch.abs(F) ** 2 / F.shape[2] / F.shape[3]  # (B, L, H, W)

        PSDs = []
        for r in self.r_range:
            mask_r = (self.R == r)  # (H, W)
            psd_vals = F[mask_r.unsqueeze(0).unsqueeze(0).repeat(F.shape[0], F.shape[1], 1, 1)].reshape(field_.shape[0], -1)  # (B, N_pixels)
            psd_vals = psd_vals.mean(dim=-1)  # (B,)
            psd_vals = 10 * torch.log10(psd_vals)
            PSDs.append(psd_vals)
        
        self.PSD_list.append(torch.stack(PSDs, dim=1).cpu())  # (B, N_r)
        self.probs_list.append(probs.cpu())
    
    def calc_result(self):
        PSD = torch.cat(self.PSD_list, dim=0)  # (num_samples, N_r)
        probs = torch.cat(self.probs_list, dim=0)  # (num_samples,)
        return PSD, probs


class RALSD(object):

    def __init__(self, ensemble_num, height=256, width=256, **fft_kwargs):
        self.ensemble_num = ensemble_num
        self.rapsd_pd = RAPSD(ensemble_num, height, width, **fft_kwargs)
        self.rapsd_gt = RAPSD(ensemble_num, height, width, **fft_kwargs)
    
    def add_test_case(self, pd, gt, probs=1):
        _, L, H, W = gt.shape
        gt_ = gt.view(self.ensemble_num, -1, L, H, W)
        mask = torch.mean(gt_[0], dim=(-2, -1)) < 0.002  # (B, L)
        mask = torch.any(mask, dim=-1)  # (B,)
        if torch.all(mask):
            return
        # else:
        #     print('mask', mask)
        self.rapsd_pd.add_test_case(pd, mask, probs)
        self.rapsd_gt.add_test_case(gt, mask, probs)
    
    def calc_result(self):
        PSD_pd, probs = self.rapsd_pd.calc_result()  # PSD_pd: (num_samples, N_r)
        PSD_gt, _ = self.rapsd_gt.calc_result()  # PSD_gt: (num_samples, N_r)
        PSD_rmse = (PSD_pd - PSD_gt) ** 2  # (num_samples, N_r)
        PSD_pd_avg = (PSD_pd * probs.unsqueeze(-1)).sum(dim=0) / probs.sum()  # (N_r,)
        PSD_gt_avg = (PSD_gt * probs.unsqueeze(-1)).sum(dim=0) / probs.sum()  # (N_r,)
        rmse_avg = (PSD_rmse * probs.unsqueeze(-1)).sum(dim=0) / probs.sum()  # (N_r,)
        lsd = torch.sqrt(rmse_avg.sum()).item()
        return PSD_pd_avg, PSD_gt_avg, lsd


class ROCAUC(object):
    
    def __init__(self, ensemble_num, level):
        self.ensemble_num = ensemble_num
        self.level = level
        self.predicts_list = []
        self.targets_list = []
        self.probs_list = []

    def add_test_case(self, pd, gt, probs=1):
        '''
        :param pd: torch.Tensor, size=(N * B, L, H, W)
        :param gt: torch.Tensor, size=(N * B, L, H, W)
        :return: None
        '''
        probs = 1 / probs

        _, L, H, W = pd.shape

        pd_ = pd.view(self.ensemble_num, -1, L, H, W)
        gt_ = gt.view(self.ensemble_num, -1, L, H, W)

        # binarize pd_ and gt_ with level
        pd_ = (pd_ >= self.level).to(torch.float32)
        gt_ = (gt_ >= self.level).to(torch.float32)

        # average over ensemble dimension
        pd_ = torch.mean(pd_, dim=0)
        gt_ = torch.mean(gt_, dim=0)

        # flatten
        pd_ = pd_.view(-1)
        gt_ = gt_.view(-1)

        # extend probs
        probs = probs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, L, H, W)  # (B, L, H, W)
        probs = probs.view(-1)

        self.predicts_list.append(pd_.cpu())
        self.targets_list.append(gt_.cpu())
        self.probs_list.append(probs.cpu())

    def calc_result(self):
        predicts_list = torch.cat(self.predicts_list, dim=0)
        targets_list = torch.cat(self.targets_list, dim=0)
        probs_list = torch.cat(self.probs_list, dim=0)

        auc = roc_auc_score(targets_list, predicts_list, sample_weight=probs_list)

        return predicts_list, targets_list, probs_list, auc


def compute_centred_coord_array(M, N):
    """Compute a 2D coordinate array, where the origin is at the center.
    Parameters
    ----------
    M : int
      The height of the array.
    N : int
      The width of the array.
    Returns
    -------
    out : ndarray
      The coordinate array.
    Examples
    --------
    # >>> compute_centred_coord_array(2, 2)
    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))

    """
    if M % 2 == 1:
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]
    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]
    YC, XC = np.ogrid[s1, s2]
    return YC, XC


def uniform_filter(X, size):
    padding_left = (size - 1) // 2
    padding_right = size - 1 - padding_left
    X = F.pad(X, (padding_left, padding_right, padding_left, padding_right), mode='constant', value=0)
    X = F.avg_pool2d(X, kernel_size=size, stride=1)
    return X


class FSS(object):

    def __init__(self, ensemble_num, level, size=5, ensemble_mode=True):
        self.ensemble_num = ensemble_num
        self.level = level
        self.size = size
        self.ensemble_mode = ensemble_mode
        
        self.fss_numer = 0.
        self.fss_denom = 0.
        self.p = 0.
        self.prob_total = 0.

    def add_test_case(self, pd, gt, probs=1):
        """
        :param pd: torch.Tensor, size=(N * B, L, H, W)
        :param gt: torch.Tensor, size=(N * B, L, H, W)
        :return: None
        """
        probs = 1 / probs
        # (N, B, L, H, W)
        pd_ = (pd >= self.level).to(torch.float)
        gt_ = (gt >= self.level).to(torch.float)
        # (B, L, H, W) to (B * L, 1, H, W)

        if self.ensemble_mode:  # Average first
            pd_ = torch.mean(pd_, dim=0)
            gt_ = torch.mean(gt_, dim=0)
        
        pd_ = uniform_filter(pd_, self.size)
        gt_ = uniform_filter(gt_, self.size)
        
        # calculate FSS with ((pd_ - gt_) ** 2) / (pd_ ** 2 + gt_ ** 2)
        numer = torch.sum((pd_ - gt_) ** 2, dim=(-1, -2, -3))  # (B * L,) or (B,)
        denom = torch.sum((pd_ ** 2 + gt_ ** 2), dim=(-1, -2, -3))  # (B * L,) or (B,)
        
        if not self.ensemble_mode:
            probs = probs.unsqueeze(0).repeat(pd_.shape[0])
        
        self.fss_numer += torch.sum(numer * probs).item()
        self.fss_denom += torch.sum(denom * probs).item()
        self.p += (pd_.mean(dim=(-1, -2, -3)) * probs).sum().item()
        self.prob_total += probs.sum().item()

        # pd_conv = pd_conv.squeeze(1).view(-1, self.length, self.height_out, self.width_out)
        # gt_conv = gt_conv.squeeze(1).view(-1, self.length, self.height_out, self.width_out)

        # self.fbs += torch.sum((pd_conv - gt_conv) **2 / prob, dim=0)
        # self.fbs_worst += torch.sum(((pd_conv**2) + (gt_conv**2) + 1e-12) / prob, dim=0)

    def calc_result(self):
        # denominator = torch.sum(self.fbs_worst, dim=(1, 2))
        # numerator = torch.sum(self.fbs, dim=(1, 2))
        # return 1 - numerator / denominator
        return 1 - self.fss_numer / self.fss_denom, self.p / self.prob_total


class MetricsEnsemble(object):

    def __init__(self, ensemble_num, height, width, eval_cfg):
        self.ensemble_num = ensemble_num
        self.height = height
        self.width = width
        self.n_test = 0

        self.crps_dict = {}
        for size, typ, stride in zip(eval_cfg.crps.sizes, eval_cfg.crps.types, eval_cfg.crps.strides):
            self.crps_dict[f"{typ}_size{size}_stride{stride}"] = CRPS(self.ensemble_num, size=size, type=typ, stride=stride)

        self.rank_histogram_dict = {}
        for noise in eval_cfg.rank_histogram.noises:
            self.rank_histogram_dict[f"noise{noise}"] = RankHistogram(self.ensemble_num, noise=noise)
        
        self.ralsd_dict = {"ralsd": RALSD(self.ensemble_num, height, width)}
        
        self.rocauc_dict = {}
        for level in eval_cfg.rocauc.levels:
            self.rocauc_dict[f"level{level}"] = ROCAUC(self.ensemble_num, level=level)
        
        self.fss_dict = {}
        for level, size in zip(eval_cfg.fss.levels, eval_cfg.fss.sizes):
            self.fss_dict[f"level{level}_size{size}"] = FSS(self.ensemble_num, level=level, size=size)
        
        self.maemse_dict = {}
        for patch_size in [1]:
            self.maemse_dict[f"patch_size{patch_size}"] = MAEMSE(self.ensemble_num, patch_size=patch_size)

        self.metric_dicts = {
            "crps": self.crps_dict,
            "rank_histogram": self.rank_histogram_dict,
            "ralsd": self.ralsd_dict,
            # "rocauc": self.rocauc_dict,
            "fss": self.fss_dict,
            "maemse": self.maemse_dict
        }

    def add_test_case(self, pd, gt, probs=1):
        self.n_test = self.n_test + 1
        for metric_dict in self.metric_dicts.values():
            for metric in metric_dict.values():
                metric.add_test_case(pd, gt, probs)

    def calc_result(self):
        result = {
            'n_test': self.n_test,
            'height': self.height,
            'width': self.width
        }

        for metric_name, metric_dict in self.metric_dicts.items():
            # print(metric_name)
            for name, metric in metric_dict.items():
                # print('!!!!!start', f"{metric_name}.{name}")
                result[f"{metric_name}.{name}"] = metric.calc_result()
            #     print(result[f"{metric_name}.{name}"])
            #     print('!!!!!end', f"{metric_name}.{name}")
            # print('------------------')
        return result


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('')
    ensemble_num = 10

    metrics = MetricsEnsemble(ensemble_num, 64, 64, 1, cfg.eval)
    
    # pd ~ uniform distribution U[0, 128]
    # gt ~ uniform distribution U[0, 128]
    B, L, H, W = 300, 1, 64, 64
    pd = torch.rand(ensemble_num * B, L, H, W) * 128
    # gt: repeat ensemble_num times, (B, L, H, W) -> (num_samples, L, H, W)
    gt = torch.rand(B, L, H, W).repeat(ensemble_num, 1, 1, 1) * 128
    probs = torch.ones(B)

    pd, gt, probs = pd.cuda(), gt.cuda(), probs.cuda()

    print(metrics.calc_result())
