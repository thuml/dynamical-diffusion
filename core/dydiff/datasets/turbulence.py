import torch
from datasets.turbulence.torch_dataset import Turbulence
from datasets.turbulence.data_module import TurbulenceDataModule
from datasets.normalizer import NullNormalizer


class TurbulenceForDyDiff(Turbulence):
    def __init__(self, root, filename, total_length=20, input_length=10, no_oracle=False, n_subsample=None):
        self.input_length = input_length
        self.no_oracle = no_oracle
        super().__init__(root, filename, total_length, n_subsample)
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data = NullNormalizer.normalize(data)  # (T, C, H, W)

        # data = data.permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        cond = data[:self.input_length]
        pred = data[self.input_length:]
        # if self.no_oracle:
        #     prev = data[self.input_length-1:self.input_length].repeat(self.total_length-self.input_length, 1, 1, 1)
        # else:
        #     prev = data[self.input_length-1:-1]
        prev = data[:self.input_length]
        cond = cond.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        pred = pred.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        prev = prev.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        # cond_all = torch.cat([cond, topography], dim=-1)
        return dict(image=pred, prev=prev, cond=cond)
    
class TurbulenceForDyDiffTest(Turbulence):
    def __init__(self, root, filename, total_length=20, input_length=10, no_oracle=False, n_subsample=None):
        self.input_length = input_length
        self.no_oracle = no_oracle
        # print(self.force_last_frame)
        super().__init__(root, filename, total_length, n_subsample)
    
    def __getitem__(self, index):
        data = super().__getitem__(index)
        data = NullNormalizer.normalize(data)  # (T, C, H, W)

        # data = data.permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        cond = data[:self.input_length]
        pred = data[self.input_length:self.total_length]
        # if self.no_oracle:
        #     prev = data[self.input_length-1:self.input_length].repeat(self.total_length-self.input_length, 1, 1, 1)
        # else:
        #     prev = data[self.input_length-1:-1]
        prev = data[:self.input_length]
        cond = cond.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        pred = pred.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        prev = prev.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        total = data.reshape(-1, 64, 64).permute(1, 2, 0).to(memory_format=torch.contiguous_format).float()
        # cond_all = torch.cat([cond, topography], dim=-1)
        return dict(image=pred, prev=prev, cond=cond, total=total)


class TurbulenceDataModuleForDyDiff(TurbulenceDataModule):
    def setup(self, stage=None):
        dataset_config = self.hparams.dataset_config
        self.train_dataset = TurbulenceForDyDiff(
            root=dataset_config.root,
            total_length=dataset_config.total_length,
            input_length=dataset_config.input_length,
            **dataset_config.train
        )
        self.test_dataset = TurbulenceForDyDiffTest(
            root=dataset_config.root,
            total_length=dataset_config.total_length,
            input_length=dataset_config.input_length,
            **dataset_config.test
        )
        print(f"Train dataset: {len(self.train_dataset)} samples")
        print(f"Test dataset: {len(self.test_dataset)} samples")
