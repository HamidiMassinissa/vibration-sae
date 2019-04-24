import torch
import torch.utils.data


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        # import pdb
        # pdb.set_trace()

        # return tuple(dataset[i] for dataset in self.datasets)
        return tuple(torch.from_numpy(dataset[i]).float()
                     for dataset in self.datasets)
