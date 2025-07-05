from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .torch_dataset import Turbulence


class TurbulenceDataModule(LightningDataModule):

    def __init__(self, dataset_config, batch_size, num_workers):
        super(TurbulenceDataModule, self).__init__()
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.save_hyperparameters()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset_config = self.hparams.dataset_config
        self.train_dataset = Turbulence(
            root=dataset_config.root,
            total_length=dataset_config.total_length,
            **dataset_config.train
        )
        self.test_dataset = Turbulence(
            root=dataset_config.root,
            total_length=dataset_config.total_length,
            **dataset_config.test
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Currently use the test set for validation. Will modify later."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
            shuffle=False
        )

    @property
    def num_train_samples(self):
        return len(self.train_dataset)

    @property
    def num_val_samples(self):
        """Currently use the test set for validation. Will modify later."""
        return len(self.test_dataset)

    @property
    def num_test_samples(self):
        return len(self.test_dataset)
