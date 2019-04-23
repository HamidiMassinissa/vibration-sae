from .datareader import DataReader
from .segmentation import Segmentation


# class Dataset(torch.utils.data.Dataset):
class Dataset():
    def __init__(self):
        self.build()

    def build(self):
        dr = DataReader()
        X = dr.data
        p = Segmentation(X)
        self.X = p.X
        self.modalities_assoc = p.modalities_assoc

    def __len__(self):
        return self.X.shape[0]

    @property
    def frame_size(self):
        return self.X.shape[1]

    @property
    def num_channels(self):
        return self.X.shape[2]

    def __getitem__(self, pos=(0, ['acc2__'])):
        """  returns one or more time series
        """
        if type(pos) is tuple:
            if len(pos) == 2:
                # dataset[idx{n|:}, channels{[s]}]
                idx, channels = pos
                idz_s = []
                for channel in channels:
                    [idz] = self.modalities_assoc[channel]
                    idz_s.append(idz)
                item = self.X[idx, :, idz_s]
                print('[Dataset.__getitem__] item.shape = {}'
                      .format(item.shape))
                # assert item.shape == (self.frame_size,) or item.shape == (len(self), self.frame_size, len(channels))
            else:
                # dataset[idx, idy, channel]
                raise ValueError(
                    'You tried dataset[idx,idy,channel]. This type of indexing is not supported')
        else:
            if type(pos) == str:
                [idz] = self.modalities_assoc[pos]
                item = self.X[:, :, idz]
                assert item.shape == (len(self), self.frame_size)
            else:
                item = self.X[pos]
                # assert item.shape == (self.frame_size, self.num_channels)  # (60, 12)

        # return torch.from_numpy(item).float()
        return item
