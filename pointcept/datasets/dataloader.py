from functools import partial
import weakref
import torch
import torch.utils.data

import pointcept.utils.comm as comm
from pointcept.datasets.utils import point_collate_fn
from pointcept.datasets import ConcatDataset
from pointcept.utils.env import set_seed


class MultiDatasetDummySampler:
    def __init__(self):
        self.dataloader = None

    def set_epoch(self, epoch):
        if comm.get_world_size() > 1:
            for dataloader in self.dataloader.dataloaders:
                dataloader.sampler.set_epoch(epoch)
        return


class MultiDatasetDataloader:
    """
    Multiple Datasets Dataloader, batch data from a same dataset and mix up ratio determined by loop of each sub dataset.
    The overall length is determined by the main dataset (first) and loop of concat dataset.
    """

    def __init__(
        self,
        concat_dataset: ConcatDataset,
        batch_size_per_gpu: int,
        num_worker_per_gpu: int,
        mix_prob=0,
        seed=None,
    ):
        self.datasets = concat_dataset.datasets
        self.ratios = [dataset.loop for dataset in self.datasets]
        # reset data loop, original loop serve as ratios
        for dataset in self.datasets:
            dataset.loop = 1
        # determine union training epoch by main dataset
        self.datasets[0].loop = concat_dataset.loop
        # build sub-dataloaders
        num_workers = num_worker_per_gpu // len(self.datasets)
        self.dataloaders = []
        for dataset_id, dataset in enumerate(self.datasets):
            if comm.get_world_size() > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None

            init_fn = (
                partial(
                    self._worker_init_fn,
                    dataset_id=dataset_id,
                    num_workers=num_workers,
                    num_datasets=len(self.datasets),
                    rank=comm.get_rank(),
                    seed=seed,
                )
                if seed is not None
                else None
            )
            self.dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size_per_gpu,
                    shuffle=(sampler is None),
                    num_workers=num_worker_per_gpu,
                    sampler=sampler,
                    collate_fn=partial(point_collate_fn, mix_prob=mix_prob),
                    pin_memory=True,
                    worker_init_fn=init_fn,
                    drop_last=True,
                    persistent_workers=True,
                )
            )
        self.sampler = MultiDatasetDummySampler()
        self.sampler.dataloader = weakref.proxy(self)

    def __iter__(self):
        iterator = [iter(dataloader) for dataloader in self.dataloaders]
        while True:
            for i in range(len(self.ratios)):
                for _ in range(self.ratios[i]):
                    try:
                        batch = next(iterator[i])
                    except StopIteration:
                        if i == 0:
                            return
                        else:
                            iterator[i] = iter(self.dataloaders[i])
                            batch = next(iterator[i])
                    yield batch

    def __len__(self):
        main_data_loader_length = len(self.dataloaders[0])
        return (
            main_data_loader_length // self.ratios[0] * sum(self.ratios)
            + main_data_loader_length % self.ratios[0]
        )

    @staticmethod
    def _worker_init_fn(worker_id, num_workers, dataset_id, num_datasets, rank, seed):
        worker_seed = (
            num_workers * num_datasets * rank
            + num_workers * dataset_id
            + worker_id
            + seed
        )
        set_seed(worker_seed)
