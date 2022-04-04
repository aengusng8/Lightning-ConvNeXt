import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from data import utils


class DInterface(pl.LightningDataModule):

    def __init__(self, args, dataset_train, dataset_val, dataset_test):
        super().__init__()
        self.args = args
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

    def setup(self, stage=None):

        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        self.sampler_train = torch.utils.data.DistributedSampler(
            self.dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=self.args.seed,
        )
        print("Sampler_train = %s" % str(self.sampler_train))
        self.sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)
        self.sampler_test = torch.utils.data.SequentialSampler(self.dataset_test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train, sampler=self.sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val, sampler=self.sampler_val,
            batch_size=int(1.5 * self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.sampler_test, sampler=self.sampler_test,
            batch_size=int(1.5 * self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=False
        )