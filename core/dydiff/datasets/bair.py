import torch
from datasets.bair.torch_dataset import BairSimpleDataset
from datasets.moving_mnist.data_module import MovingMNISTDataModule
from datasets.normalizer import RGBNormalizer

from einops import rearrange

class BairForDyDiff(BairSimpleDataset):
    def __init__(self, base_path, total_length, input_length, no_oracle=False, n_subsample=None, train=True, last_only=False):
        super().__init__(base_path, train, total_length)
        self.total_length = total_length
        self.input_length = input_length
        self.no_oracle = no_oracle
        self.last_only = last_only
        self.num_samples = super().__len__()
        if n_subsample is not None:
            self.num_samples = min(self.num_samples, n_subsample)
        self.indices = torch.randperm(self.num_samples, generator=torch.Generator().manual_seed(42)).tolist()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        imgs, actions, states = super().__getitem__(self.indices[index])  # (T, 1, H, W, C)
        imgs = imgs / 255.0
        imgs = torch.from_numpy(imgs)
        imgs = RGBNormalizer.normalize(imgs)
        imgs = rearrange(imgs, 'T H W C -> T C H W')

        # data = data.permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        cond = imgs[:self.input_length]
        pred = imgs[self.input_length:]
        # if self.no_oracle:
        #     prev = imgs[self.input_length-1:self.input_length].repeat(self.total_length-self.input_length, 1, 1, 1)
        # else:
        #     prev = imgs[self.input_length-1:-1]
        if self.last_only:
            prev = imgs[self.input_length-1:self.input_length]
        else:
            prev = imgs[:self.input_length]
        cond = cond.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        pred = pred.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        prev = prev.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        total = imgs.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        # cond_all = torch.cat([cond, topography], dim=-1)
        T, C = actions.shape
        actions_pad = torch.zeros(T+1, C)
        actions_pad[1:] = torch.from_numpy(actions)
        return dict(image=pred, prev=prev, cond=cond, total=total, actions=actions_pad)
    
    
class BairDataModuleForDyDiff(MovingMNISTDataModule):
    def setup(self, stage=None):
        dataset_config = self.hparams.dataset_config
        self.train_dataset = BairForDyDiff(
            base_path=dataset_config.base_path,
            total_length=dataset_config.total_length,
            input_length=dataset_config.input_length,
            train=True,
            **dataset_config.train
        )
        self.test_dataset = BairForDyDiff(
            base_path=dataset_config.base_path,
            total_length=dataset_config.total_length,
            input_length=dataset_config.input_length,
            train=False,
            **dataset_config.test
        )

