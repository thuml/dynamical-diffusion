import numpy as np
import torch
import torch.nn.functional as F
import time


class per_grid_cell_csi(object):
    def __init__(self, length, height, width, level):
        self.TP = torch.zeros((length, height, width))
        self.FP = torch.zeros((length, height, width))
        self.FN = torch.zeros((length, height, width))
        self.level = level

    def add_test_case(self, pd, gt, prob=1):
        '''
        :param pd: torch.Tensor, size=(B, L, H, W)
        :param gt: torch.Tensor, size=(B, L, H, W)
        :param prob: torch.Tensor, size=(B)
        :return: None
        '''
        # prob (torch.Tensor): size(B)
        pd_ = (pd >= self.level).to(torch.float)
        gt_ = (gt >= self.level).to(torch.float)
        # print(pd_.device, gt_.device, prob.device)
        # print('shape: ', pd_.shape, gt_.shape, self.TP.shape, prob)
        self.TP += torch.sum(pd_ * gt_ / prob, dim=0)
        self.FP += torch.sum(pd_ * (1 - gt_) / prob, dim=0)
        self.FN += torch.sum((1 - pd_) * gt_ / prob, dim=0)

    def cal_result(self):
        denominator = torch.sum(self.TP + self.FP + self.FN, dim=(1, 2))
        numerator = torch.sum(self.TP, dim=(1, 2))
        return numerator / denominator

    def to(self, device):
        self.TP = self.TP.to(device)
        self.FP = self.FP.to(device)
        self.FN = self.FN.to(device)

class per_grid_cell_csi_v2(object):
    def __init__(self, length, height, width, level):
        self.TP = torch.zeros((length, height, width))
        self.FP = torch.zeros((length, height, width))
        self.FN = torch.zeros((length, height, width))
        self.level = level

    def add_test_case(self, pd, gt, prob=1):
        '''
        :param pd: torch.Tensor, size=(B, L, H, W)
        :param gt: torch.Tensor, size=(B, L, H, W)
        :param prob: torch.Tensor, size=(B)
        :return: None
        '''
        # prob (torch.Tensor): size(B)
        pd_ = (pd >= self.level).to(torch.float)
        gt_ = (gt >= self.level).to(torch.float)

        self.TP += torch.sum(pd_ * gt_ / prob, dim=0)
        self.FP += torch.sum(pd_ * (1 - gt_) / prob, dim=0)
        self.FN += torch.sum((1 - pd_) * gt_ / prob, dim=0)

    def cal_result(self):
        denominator = self.TP + self.FP + self.FN
        denominator[denominator == 0] = -1
        result = self.TP / denominator
        result[result < 0] = 0
        result = torch.mean(result, dim=(1, 2))
        return result

    def to(self, device):
        self.TP = self.TP.to(device)
        self.FP = self.FP.to(device)
        self.FN = self.FN.to(device)



class per_grid_cell_window(object):
    def __init__(self, length, height, width, level, size=5):
        self.size=size
        height_pool = (height + 2 * (size//2) - size) // (size//2) + 1
        width_pool = (width + 2 * (size//2) - size) // (size//2) + 1
        self.TP = torch.zeros((length, height_pool, width_pool))
        self.FP = torch.zeros((length, height_pool, width_pool))
        self.FN = torch.zeros((length, height_pool, width_pool))
        self.height = height
        self.length = length
        self.width = width
        self.level = level
        self.weight = torch.ones(1, 1, self.size, self.size)

    def add_test_case(self, pd, gt, prob=1):
        # print(pd.shape, gt.shape)
        pd_ = (pd >= self.level).to(torch.float)
        gt_ = (gt >= self.level).to(torch.float)
        all = ((pd_ + gt_) >= 1).to(torch.float)

        pd_pool = F.max_pool2d(pd_, kernel_size=self.size, stride=self.size//2, padding=self.size//2)
        gt_pool = F.max_pool2d(gt_, kernel_size=self.size, stride=self.size//2, padding=self.size//2)

        self.TP += torch.sum(pd_pool * gt_pool / prob, dim=0)
        self.FP += torch.sum(pd_pool * (1 - gt_pool) / prob, dim=0)
        self.FN += torch.sum((1 - pd_pool) * gt_pool / prob, dim=0)

    def cal_result(self):
        denominator = torch.sum(self.TP + self.FP + self.FN, dim=(1, 2))
        numerator = torch.sum(self.TP, dim=(1, 2))
        return numerator / denominator

    def to(self, device):
        self.TP = self.TP.to(device)
        self.FP = self.FP.to(device)
        self.FN = self.FN.to(device)
        self.weight = self.weight.to(device)


class per_grid_cell_window7(per_grid_cell_window):
    def __init__(self, length, height, width, level):
        super(per_grid_cell_window7, self).__init__(length, height, width, level, size=7)

class per_grid_cell_window5(per_grid_cell_window):
    def __init__(self, length, height, width, level):
        super(per_grid_cell_window5, self).__init__(length, height, width, level, size=5)

class per_grid_cell_window3(per_grid_cell_window):
    def __init__(self, length, height, width, level):
        super(per_grid_cell_window3, self).__init__(length, height, width, level, size=3)


class fss(object):
    def __init__(self, length, height, width, level, size=5):
        self.size=size
        self.pad = self.size // 2
        self.stride = int((self.size + 2) // 4)

        self.height_out = int((height + 2 * self.pad - self.size) // self.stride + 1)
        self.width_out = int((width + 2 * self.pad - self.size) // self.stride + 1)
        self.fbs = torch.zeros((length, self.height_out, self.width_out))
        self.fbs_worst = torch.zeros((length, self.height_out, self.width_out))
        self.height = height
        self.length = length
        self.width = width
        self.level = level
        self.weight = torch.ones(1, 1, self.size, self.size) / (self.size ** 2)

    def add_test_case(self, pd, gt, prob=1):
        # print(pd.shape, gt.shape)
        pd_ = (pd >= self.level).to(torch.float)
        gt_ = (gt >= self.level).to(torch.float)
        all = ((pd_ + gt_) >= 1).to(torch.float)
        # (B, L, H, W) to (B * L, 1, H, W)
        pd_conv = pd_.unsqueeze(2).flatten(0, 1)
        gt_conv = gt_.unsqueeze(2).flatten(0, 1)
        pd_conv = F.conv2d(pd_conv, self.weight, padding=self.pad, stride=self.stride)
        gt_conv = F.conv2d(gt_conv, self.weight, padding=self.pad, stride=self.stride)
        # (B * L, 1, H, W) to (B, L, H, W)

        pd_conv = pd_conv.squeeze(1).view(-1, self.length, self.height_out, self.width_out)
        gt_conv = gt_conv.squeeze(1).view(-1, self.length, self.height_out, self.width_out)

        self.fbs += torch.sum((pd_conv - gt_conv) **2 / prob, dim=0)
        self.fbs_worst += torch.sum(((pd_conv**2) + (gt_conv**2) + 1e-12) / prob, dim=0)

    def cal_result(self):
        denominator = torch.sum(self.fbs_worst, dim=(1, 2))
        numerator = torch.sum(self.fbs, dim=(1, 2))
        return 1 - numerator / denominator

    def to(self, device):
        self.fbs = self.fbs.to(device)
        self.fbs_worst = self.fbs_worst.to(device)
        self.weight = self.weight.to(device)


class fss_window7(fss):
    def __init__(self, length, height, width, level):
        super(fss_window7, self).__init__(length, height, width, level, size=7)

class fss_window5(fss):
    def __init__(self, length, height, width, level):
        super(fss_window5, self).__init__(length, height, width, level, size=5)

class fss_window3(fss):
    def __init__(self, length, height, width, level):
        super(fss_window3, self).__init__(length, height, width, level, size=3)


class interval_csi(object):
    def __init__(self, length, height, width, level):
        self.TP = torch.zeros((length, height, width))
        self.FP = torch.zeros((length, height, width))
        self.FN = torch.zeros((length, height, width))
        self.level = level

    def add_test_case(self, pd, gt, prob=1):
        '''
        :param pd: torch.Tensor, size=(B, L, H, W)
        :param gt: torch.Tensor, size=(B, L, H, W)
        :param prob: torch.Tensor, size=(B)
        :return: None
        '''
        # prob (torch.Tensor): size(B)
        pd_ = (pd >= self.level[0]) * (pd < self.level[1])
        gt_ = (gt >= self.level[0]) * (gt < self.level[1])

        pd_ = pd_.to(torch.float)
        gt_ = gt_.to(torch.float)

        self.TP += torch.sum(pd_ * gt_ / prob, dim=0)
        self.FP += torch.sum(pd_ * (1 - gt_) / prob, dim=0)
        self.FN += torch.sum((1 - pd_) * gt_ / prob, dim=0)

    def cal_result(self):
        denominator = torch.sum(self.TP + self.FP + self.FN, dim=(1, 2))
        numerator = torch.sum(self.TP, dim=(1, 2))
        return numerator / denominator

    def to(self, device):
        self.TP = self.TP.to(device)
        self.FP = self.FP.to(device)
        self.FN = self.FN.to(device)


class interval_csi_window(object):
    def __init__(self, length, height, width, level):
        self.size=5 # should be odd number
        self.TP = torch.zeros((length, height, width))
        self.FP = torch.zeros((length, height, width))
        self.FN = torch.zeros((length, height, width))
        self.height = height
        self.length = length
        self.width = width
        self.level = level
        self.weight = torch.ones(1, 1, self.size, self.size)

    def add_test_case(self, pd, gt, prob=1):
        # print(pd.shape, gt.shape)
        pd_ = (pd >= self.level[0]) * (pd < self.level[1])
        gt_ = (gt >= self.level[0]) * (gt < self.level[1])

        pd_ = pd_.to(torch.float)
        gt_ = gt_.to(torch.float)

        all = ((pd_ + gt_) >= 1).to(torch.float)
        # (B, L, H, W) to (B * L, 1, H, W)
        pd_conv = pd_.unsqueeze(2).flatten(0, 1)
        gt_conv = gt_.unsqueeze(2).flatten(0, 1)
        pd_conv = F.conv2d(pd_conv, self.weight, padding=self.size // 2)
        gt_conv = F.conv2d(gt_conv, self.weight, padding=self.size // 2).squeeze(1)
        # (B * L, 1, H, W) to (B, L, H, W)
        pd_conv = pd_conv.squeeze(1).view(-1, self.length, self.height, self.width)
        gt_conv = gt_conv.squeeze(1).view(-1, self.length, self.height, self.width)
        gt_hit = (pd_conv * gt_ > 0).to(torch.float)
        pd_hit = (gt_conv * pd_ > 0).to(torch.float)
        self.TP += torch.sum((all - (pd_ - pd_hit) - (gt_ - gt_hit)) / prob, dim=0)
        self.FP += torch.sum((pd_ - pd_hit) / prob, dim=0)
        self.FN += torch.sum((gt_ - gt_hit) / prob, dim=0)

    def cal_result(self):
        denominator = torch.sum(self.TP + self.FP + self.FN, dim=(1, 2))
        numerator = torch.sum(self.TP, dim=(1, 2))
        return numerator / denominator

    def to(self, device):
        self.TP = self.TP.to(device)
        self.FP = self.FP.to(device)
        self.FN = self.FN.to(device)
        self.weight = self.weight.to(device)

#
class LP_metric(object):
    def __init__(self, ensemble_num, LP, LP2):
        self.ensemble_num = ensemble_num
        self.l1_loss_list = []
        self.prob_list = []
        self.LP = LP
        self.LP2 = LP2

    def add_test_case(self, pd, gt, prob=1):
        '''
        :param pd: torch.Tensor, size=(N * B, L, H, W)
        :param gt: torch.Tensor, size=(N * B , L, H, W)
        :param prob: torch.Tensor, size=(B, 1, 1, 1)
        :return: None
        '''
        # prob (torch.Tensor): size(B)
        _, L, H, W = pd.size()

        if type(prob) == torch.Tensor:
            prob = prob.unsqueeze(0)
        pd_ = pd.view(self.ensemble_num, -1, L, H, W)
        gt_ = gt.view(self.ensemble_num, -1, L, H, W)
        # mask = gt_ >= 0
        mask = torch.ones_like(gt_)
        l1_loss = torch.sum((self.LP(pd_ - gt_) * mask), dim=(0,3,4)) / \
                  torch.sum(mask, dim=(0,3,4))
        l1_loss = self.LP2(l1_loss)
        self.l1_loss_list.append(l1_loss.cpu())
        if type(prob) == torch.Tensor:
            self.prob_list.extend((1 / prob).view(-1).cpu().tolist())
        else:
            self.prob_list.extend(torch.tensor([1 / prob for i in range(gt_.size(1))]))


    def cal_result(self):

        crps_list = torch.cat(self.l1_loss_list, dim=0)
        probs = torch.Tensor(self.prob_list).view(-1, 1)
        # print('crps', crps_list)
        # print('probs', probs)
        nan_ids = torch.max(torch.isnan(crps_list), dim=1).values.to(torch.float32) == 0
        crps_list = crps_list[nan_ids]
        probs = probs[nan_ids]

        crps = torch.sum((crps_list * probs) / torch.sum(probs), dim=0)
        return crps


    def to(self, device):
        pass


class MSE(LP_metric):
    def __init__(self, ensemble_num):
        super(MSE, self).__init__(ensemble_num, lambda x: x ** 2, lambda x: torch.sqrt(x))

class PSNR(LP_metric):
    def __init__(self, ensemble_num):
        super(PSNR, self).__init__(ensemble_num, lambda x: x ** 2, lambda x: 20 * torch.log(1 / torch.sqrt(x)))


class CRPS_PWM(object):
    def __init__(self, ensemble_num):
        self.ensemble_num = ensemble_num
        self.crps_list = []
        self.prob_list = []

    def add_test_case(self, pd, gt, prob=1):
        '''
        :param pd: torch.Tensor, size=(N * B, L, H, W)
        :param gt: torch.Tensor, size=(N * B , L, H, W)
        :param prob: torch.Tensor, size=(1, 1, 1, B)
        :return: None
        '''
        # prob (torch.Tensor): size(B)
        _, L, H, W = pd.size()

        if type(prob) == torch.Tensor:
            prob = prob.unsqueeze(0)
        pd_ = pd.view(self.ensemble_num, -1, L, H, W)
        gt_ = gt.view(self.ensemble_num, -1, L, H, W)
        mask = (- gt_[0]) <= 0
        # l1_loss = torch.sum((torch.abs(pd_ - gt_) * mask), dim=(0,3,4)) / \
        #     torch.sum(mask , dim=(0,2,3,4))

        l1_loss = torch.mean(torch.abs(pd_ - gt_), dim=0)

        if type(prob) == torch.Tensor:
            self.prob_list.extend(prob.view(-1).cpu().tolist())
        else:
            self.prob_list.extend([prob for i in range(gt_.size(1))])

        # print(temp_l1_loss.size(), torch.sum((torch.abs(gt_[0] - pd_[1]) * mask[0]), dim=(1, 2, 3)).size())
        if self.ensemble_num > 1:
            beta0 = torch.mean(pd_, dim=0)
            pd_sorted, _ = torch.sort(pd_, dim=0, descending=False)
            weight = torch.arange(0, self.ensemble_num).view(-1, 1, 1, 1, 1).to(pd_sorted.device)
            # print(pd_sorted.size(), weight.size())
            beta1 = torch.sum(weight * pd_sorted, dim=0) / (self.ensemble_num * (self.ensemble_num - 1))
            #print(torch.mean(beta0, dim=(2, 3)), torch.mean(beta1, dim=(2, 3)), beta0 - 2 * beta1)
            crps_pixel = l1_loss + beta0 - 2 * beta1
        else:
            crps_pixel = l1_loss
        # print(crps_pixel.size(), mask.size())

        crps = torch.sum(crps_pixel * mask, dim=(2, 3)) / torch.sum(mask, dim=(2, 3))
        self.crps_list.append(crps.cpu())


    def cal_result(self):
        crps_list = torch.cat(self.crps_list, dim=0)
        probs = torch.Tensor(self.prob_list).view(-1, 1)
        # print('crps', crps_list)
        # print('probs', probs)
        nan_ids = torch.max(torch.isnan(crps_list), dim=1).values.to(torch.float32) == 0
        crps_list = crps_list[nan_ids]
        probs = probs[nan_ids]

        crps = torch.sum((crps_list * probs) / torch.sum(probs), dim=0)
        return crps

    def to(self, device):
        pass

class L1(CRPS_PWM):
    def __init__(self, ensemble_num):
        super(L1, self).__init__(1)


class Pooled_CRPS_PWM(object):
    def __init__(self, ensemble_num, size=4, pool_type='avg'):
        self.size=size
        self.stride = int((self.size + 2) // 4)
        self.weight = torch.ones(1, 1, self.size, self.size) / (self.size ** 2)
        self.ensemble_num = ensemble_num
        self.crps_list = []
        self.prob_list = []
        if pool_type == 'avg':
            self.pool_layer = F.avg_pool2d
        elif pool_type == 'max':
            self.pool_layer = F.max_pool2d
        elif pool_type == 'min':
            self.pool_layer = lambda x, kernel_size, stride: - F.max_pool2d(-x, kernel_size, stride)

    def add_test_case(self, pd, gt, prob=1):
        '''
        :param pd: torch.Tensor, size=(N * B, L, H, W)
        :param gt: torch.Tensor, size=(N * B , L, H, W)
        :param prob: torch.Tensor, size=(1, 1, 1, B)
        :return: None
        '''
        # prob (torch.Tensor): size(B)
        _, L, H, W = pd.size()

        if type(prob) == torch.Tensor:
            prob = prob.unsqueeze(0)
        mask = F.max_pool2d(- gt.view(self.ensemble_num, -1, L, H, W)[0], kernel_size= self.size, stride=self.stride) <= 0
        pd = pd.unsqueeze(2).flatten(0, 1)
        gt = gt.unsqueeze(2).flatten(0, 1)

        pd = self.pool_layer(pd, kernel_size=self.size, stride=self.stride)
        gt = self.pool_layer(gt,  kernel_size=self.size, stride=self.stride)
        _, _, H_, W_ = gt.size()
        pd = pd.squeeze(1).view(-1, L, H_, W_)
        gt = gt.squeeze(1).view(-1, L, H_, W_)

        pd_ = pd.view(self.ensemble_num, -1, L, H_, W_)
        gt_ = gt.view(self.ensemble_num, -1, L, H_, W_)
        # l1_loss = torch.sum((torch.abs(pd_ - gt_) * mask), dim=(0,3,4)) / \
        #     torch.sum(mask , dim=(0,2,3,4))

        l1_loss = torch.mean(torch.abs(pd_ - gt_), dim=0)

        if type(prob) == torch.Tensor:
            self.prob_list.extend((1/prob).view(-1).cpu().tolist())
        else:
            self.prob_list.extend([1 / prob for i in range(gt_.size(1))])

        # print(temp_l1_loss.size(), torch.sum((torch.abs(gt_[0] - pd_[1]) * mask[0]), dim=(1, 2, 3)).size())
        if self.ensemble_num > 1:
            beta0 = torch.mean(pd_, dim=0)
            pd_sorted, _ = torch.sort(pd_, dim=0, descending=False)
            weight = torch.arange(0, self.ensemble_num).view(-1, 1, 1, 1, 1).to(pd_sorted.device)
            beta1 = torch.sum(weight * pd_sorted, dim=0) / (self.ensemble_num * (self.ensemble_num - 1))
            crps_pixel = l1_loss + beta0 - 2 * beta1
            # print(torch.mean( beta0 - 2 * beta1))
        else:
            crps_pixel = l1_loss
        # print(crps_pixel.size(), mask.size())
        crps = torch.sum(crps_pixel * mask, dim = (2,3)) / torch.sum(mask, dim = (2,3))
        self.crps_list.append(crps.cpu())


    def cal_result(self):
        crps_list = torch.cat(self.crps_list, dim=0)
        probs = torch.Tensor(self.prob_list).view(-1, 1)
        nan_ids = torch.max(torch.isnan(crps_list), dim=1).values.to(torch.float32) == 0
        crps_list = crps_list[nan_ids]
        probs = probs[nan_ids]
        crps = torch.sum((crps_list * probs) / torch.sum(probs), dim=0)
        return crps

    def to(self, device):
        self.weight = self.weight.to(device)

class Pooled_CRPS_PWM_w2avg(Pooled_CRPS_PWM):
    def __init__(self, ensemble_num):
        super(Pooled_CRPS_PWM_w2avg, self).__init__(ensemble_num, size=2, pool_type='avg')

class Pooled_CRPS_PWM_w4avg(Pooled_CRPS_PWM):
    def __init__(self, ensemble_num):
        super(Pooled_CRPS_PWM_w4avg, self).__init__(ensemble_num, size=4, pool_type='avg')

class Pooled_CRPS_PWM_w8avg(Pooled_CRPS_PWM):
    def __init__(self, ensemble_num):
        super(Pooled_CRPS_PWM_w8avg, self).__init__(ensemble_num, size=8, pool_type='avg')

class Pooled_CRPS_PWM_w16avg(Pooled_CRPS_PWM):
    def __init__(self, ensemble_num):
        super(Pooled_CRPS_PWM_w16avg, self).__init__(ensemble_num, size=16, pool_type='avg')

class Pooled_CRPS_PWM_w2max(Pooled_CRPS_PWM):
    def __init__(self, ensemble_num):
        super(Pooled_CRPS_PWM_w2max, self).__init__(ensemble_num, size=2, pool_type='max')

class Pooled_CRPS_PWM_w4max(Pooled_CRPS_PWM):
    def __init__(self, ensemble_num):
        super(Pooled_CRPS_PWM_w4max, self).__init__(ensemble_num, size=4, pool_type='max')

class Pooled_CRPS_PWM_w8max(Pooled_CRPS_PWM):
    def __init__(self, ensemble_num):
        super(Pooled_CRPS_PWM_w8max, self).__init__(ensemble_num, size=8, pool_type='max')

class Pooled_CRPS_PWM_w16max(Pooled_CRPS_PWM):
    def __init__(self, ensemble_num):
        super(Pooled_CRPS_PWM_w16max, self).__init__(ensemble_num, size=16, pool_type='max')


class economic(object):
    def __init__(self, length, height, width, level, duration):
        self.TP = torch.zeros((height // 4, width // 4))
        self.FP = torch.zeros((height // 4, width // 4))
        self.FN = torch.zeros((height // 4, width // 4))
        self.TN = torch.zeros((height // 4, width // 4))
        self.level = level
        self.duration = duration

    def add_test_case(self, pd, gt, prob=1):
        '''
        :param pd: torch.Tensor, size=(B, L, H, W)
        :param gt: torch.Tensor, size=(B, L, H, W)
        :param prob: torch.Tensor, size=(B, 1, 1, 1)
        :return: None
        '''
        # prob (torch.Tensor): size(B)
        pd_ = torch.mean(pd, dim=1) * self.duration
        gt_ = torch.mean(gt, dim=1) * self.duration

        pd_ = F.avg_pool2d(pd_, kernel_size=4)
        gt_ = F.avg_pool2d(gt_, kernel_size=4)

        pd_ = (pd_ >= self.level).to(torch.float)
        gt_ = (gt_ >= self.level).to(torch.float)

        self.TP += torch.sum(pd_ * gt_ / prob, dim=0)
        self.FP += torch.sum(pd_ * (1 - gt_) / prob, dim=0)
        self.FN += torch.sum((1 - pd_) * gt_ / prob, dim=0)
        self.TN += torch.sum((1 - pd_) * (1 - gt_) / prob, dim=0)

    def cal_result(self):
        eco_list = []
        num = 10
        acc_tp = self.TP
        acc_fp = self.FP
        acc_fn = self.FN
        acc_tn = self.TN
        for ratio in range(num + 1):
            ratio = ratio / num
            denominator = torch.sum(acc_tp + acc_fp + acc_fn + acc_tn)
            numerator = torch.sum((acc_tp + acc_fp) * ratio + acc_fn)
            Ef = numerator / denominator
            pc = torch.sum(acc_tp + acc_fn)/ denominator
            if pc > ratio:
                Ec = ratio
            else:
                Ec = pc
            Ep = pc * ratio
            if Ep-Ec <= 0:
                eco_list.append(torch.tensor(0).to(Ef.device))
            elif (Ef-Ec) / (Ep-Ec) < 0:
                eco_list.append(torch.tensor(0).to(Ef.device))
            else:
                eco_list.append((Ef-Ec) / (Ep-Ec))

        return torch.Tensor(eco_list)

    def to(self, device):
        self.TP = self.TP.to(device)
        self.FP = self.FP.to(device)
        self.FN = self.FN.to(device)
        self.TN = self.TN.to(device)

class economic_90min(economic):
    def __init__(self, length, height, width, level):
        super(economic_90min, self).__init__(length, height, width, level, 1.5)

class economic_3h(economic):
    def __init__(self, length, height, width, level):
        super(economic_3h, self).__init__(length, height, width, level, 3)



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

class rapsd(object):
    def __init__(self, height=256, width=256, return_freq=False, d=1.0, normalize=False, **fft_kwargs):
        self.return_freq = return_freq
        YC, XC = compute_centred_coord_array(height, width)
        self.R = torch.tensor(np.sqrt(XC * XC + YC * YC).round())# .cuda()
        self.L = max(height, width)
        if self.L % 2 == 0:
            r_range = np.arange(0, int(self.L / 2) + 1)
        else:
            r_range = np.arange(0, int(self.L / 2))
        self.r_range = r_range
        self.normalize = normalize
        self.d = d

    def __call__(self, Z, prob=1):
        # if len(Z.shape) != 2:
        #     raise ValueError(
        #         f"{len(Z.shape)} dimensions are found, but the number "
        #         "of dimensions should be 2"
        #     )
        #
        # if np.sum(np.isnan(Z)) > 0:
        #     raise ValueError("input array Z should not contain nans")

        F = torch.fft.fftshift(torch.fft.fft2(Z), dim=[-1, -2])
        F = torch.abs(F) ** 2 / F.shape[2] / F.shape[3]
        # print(F.shape,prob.shape)
        F = F / prob

        result = []
        # for l in range(0,F.shape[1]):
        #     result_l = []
        #     for r in self.r_range:
        #         MASK = self.R == r
        #         F_vals = F[:,l][MASK.unsqueeze(0).repeat(1,1,1)]
        #         result_l.append(F_vals.sum().unsqueeze(dim=0))
        #     result_l = torch.cat(result_l, dim=0)
        #     result.append(result_l.unsqueeze(dim=0))
        for r in self.r_range:
            MASK = self.R == r
            F_vals = F[MASK.unsqueeze(0).unsqueeze(0).repeat(F.shape[0], F.shape[1], 1, 1)]
            F_vals = F_vals.reshape(F.shape[0], F.shape[1], -1).mean(dim=[-1]).sum(dim=0)
            result.append(F_vals.unsqueeze(1))

        result = torch.cat(result, dim=1)

        if self.normalize:
            result /= np.sum(result)

        if self.return_freq:
            freq = np.fft.fftfreq(self.L, d=self.d)
            freq = freq[self.r_range]
            return result, freq
        else:
            return result

    def to(self, device):
        self.R = self.R.to(device)

class Metrics(object):
    '''
    :param gt: np.ndarray, shape = (batch, pred_length, height, width)
    :param pd: np.ndarray, shape = (batch, pred_length, height, width)
    :return: dict, containing various metrics (csi, rapsd)
    '''
    def __init__(self, height, width, length, device, levels, ensemble_num=1):
        # self.csi_ = [27.6, 40.3, 52.9, 72.3]
        self.csi_ = levels
        self.acc_csi_ = [10 * (i+1) for i in range(6)]
        self.cnt = 0
        self.height = height
        self.width = width
        self.rapsd_pd = None
        self.rapsd_gt = None
        self.rapsd = rapsd(height, width)
        print(device)
        self.rapsd.to(device)

        self.csi_metrics = {
            'per_grid_cell_csi': per_grid_cell_csi,
            'per_grid_cell_window7': per_grid_cell_window7,
            'per_grid_cell_window5': per_grid_cell_window5,
            'per_grid_cell_window3': per_grid_cell_window3,
            'fss_window7': fss_window7,
            'fss_window5': fss_window5,
            'fss_window3': fss_window3,
        }
        self.ensemble_metrics = {
            'l1': L1,
            'MSE': MSE,
            'crps_pwm': CRPS_PWM,
            'PSNR':PSNR,
            'PSNR_dBZ':PSNR,
            # 'l1_dBZ':l1_dBZ,
            # 'MSE_dBZ': MSE_dBZ,
            'Pooled_CRPS_PWM_w2avg': Pooled_CRPS_PWM_w2avg,
            'Pooled_CRPS_PWM_w8avg': Pooled_CRPS_PWM_w8avg,
            'Pooled_CRPS_PWM_w2max': Pooled_CRPS_PWM_w2max,
            'Pooled_CRPS_PWM_w8max': Pooled_CRPS_PWM_w8max,
        }
        self.metrics_names = []
        for metric in self.csi_metrics:
            for i, level in enumerate(self.csi_):
                temp_metric = self.csi_metrics[metric](length, height, width, level=level)
                temp_name = '{}_list_{}'.format(metric, i)
                temp_metric.to(device)
                setattr(self, temp_name, temp_metric)
                self.metrics_names.append(temp_name)

        for metric in self.ensemble_metrics:
            temp_metric = self.ensemble_metrics[metric](ensemble_num)
            temp_name = '{}_list'.format(metric)
            temp_metric.to(device)
            setattr(self, temp_name, temp_metric)
            self.metrics_names.append(temp_name)

        self.length=length

    def add_test_case(self, pd, gt, prob=1):
        # if type(prob) == int:
        #     prob = torch.Tensor((prob,)).to(pd.device)
        for m in self.metrics_names:
            getattr(self, m).add_test_case(pd, gt, prob)

        if self.cnt == 0:
            self.rapsd_gt = self.rapsd(gt)
            self.rapsd_pd = self.rapsd(pd)
        else:
            self.rapsd_gt += self.rapsd(gt)
            self.rapsd_pd += self.rapsd(pd)
        self.cnt = self.cnt + pd.shape[0]


    def cal(self):
        result = {}
        for m in self.metrics_names:
            result[m] = getattr(self, m).cal_result().cpu()
            result[f'{m}_mean'] = result[m].mean()

        rapsd_pd = self.rapsd_pd.cpu()
        rapsd_gt = self.rapsd_gt.cpu()

        result['rapsd_pd'] = rapsd_pd / self.cnt
        result['rapsd_gt'] = rapsd_gt / self.cnt
        result['cnt'] = self.cnt
        result['height'] = self.height
        result['width'] = self.width
        return result



class metrics_psd(object):
    '''
    :param gt: np.ndarray, shape = (batch, pred_length, height, width)
    :param pd: np.ndarray, shape = (batch, pred_length, height, width)
    :return: dict, containing various metrics (csi, rapsd)
    '''
    def __init__(self, height, width, length, device, levels, ensemble_num=1):
        # self.csi_ = [27.6, 40.3, 52.9, 72.3]
        self.csi_ = levels
        self.acc_csi_ = [5 * (i+1) for i in range(4)]
        self.cnt = 0
        self.height = height
        self.width = width
        self.length=length
        self.rapsd_pd = None
        self.rapsd_gt = None
        self.rapsd = rapsd(height, width)
        print(device)
        self.rapsd.to(device)
        self.sum_prob = 0

    def add_test_case(self, pd, gt, prob=1):
        if type(prob) == int:
            self.sum_prob += prob
            prob = torch.Tensor((prob,)).to(pd.device)
        self.sum_prob += (1 / prob).sum()
        if self.cnt == 0:
            self.rapsd_gt = self.rapsd(gt, prob.view(-1, 1, 1, 1))
            self.rapsd_pd = self.rapsd(pd, prob.view(-1, 1, 1, 1))
        else:
            self.rapsd_gt += self.rapsd(gt, prob.view(-1, 1, 1, 1))
            self.rapsd_pd += self.rapsd(pd, prob.view(-1, 1, 1, 1))
        self.cnt = self.cnt + pd.shape[0]

    def cal(self):
        result = {}

        self.rapsd_pd = self.rapsd_pd.cpu()
        self.rapsd_gt = self.rapsd_gt.cpu()

        result['rapsd_pd'] = self.rapsd_pd / self.cnt
        result['rapsd_gt'] = self.rapsd_gt / self.cnt
        result['cnt'] = self.cnt
        result['height'] = self.height
        result['width'] = self.width
        return result