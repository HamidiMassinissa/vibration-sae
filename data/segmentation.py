import os
import sys
import numpy as np
import pickle
from sklearn.feature_extraction.image import extract_patches

from .datareader import DataReader
from config import Configuration as config


# Segmentation : (Data -> seq_len -> overlap) -> learning_dataset_format
class Segmentation(object):
    def __init__(self, data, seq_len=config.framesize, overlap=config.overlap):
        self.seq_len = seq_len
        self.step = int(seq_len * (1 - overlap))

        if self._loaded():
            # 1. determine shape
            shape = self._get_shape(data)

            # 2.a. load files
            X, modalities_assoc = \
                self._load_mmap_file(shape, create=False)
        else:
            # 2.b. create files and build lds, modalities_assoc
            X, modalities_assoc = \
                self._build_learning_dataset(data)

        # 3. return them or store them as class attributes
        self._X = X
        self._modalities_assoc = modalities_assoc

    def _loaded(self):
        files = [
            'learning_dataset-',
            'modalities_assoc-'
        ]

        check = True

        for f in files:
            # check existence
            filename = \
                f + \
                config.VERSION + '.' + \
                config.REVISION + \
                '.mmap'
            dest = os.path.join(config.EXPERIMENTSFOLDER, filename)
            # dest = os.path.join('generated', filename)

            check = check and os.path.exists(dest)

        print('check = %s -----------------------------------------' % (check,))
        return check

    @property
    def num_features(self):
        """ number of features that are outputted from this pipeline """
        return self._X.shape[2]

    @property
    def X(self):
        return self._X

    @property
    def modalities_assoc(self):
        """
         Returns an associative list or a dictionry which will correspond
         each modality to a list of positions in the tensor X.
        """
        return self._modalities_assoc

    def _load_mmap_file(self, shape, create=False):
        """
        Synopsis
         Allocate memory for learning_dataset as well as for modalities assoc
         and returns the filenames associated with these memory chunks.

         Returns
        """
        num_sequences, num_samples, num_features = shape

        mode = 'w+' if create else 'r+'

        # lds: learning dataset
        filename = \
            'learning_dataset-' + \
            config.VERSION + '.' + \
            config.REVISION + \
            '.mmap'
        dest = os.path.join(config.EXPERIMENTSFOLDER, filename)
        # dest = os.path.join('generated', filename)

        # build mmap file from scratch, It is REQUIRED!
        # print('Building from scratch %s ...' % dest)
        mmap = np.memmap(
            dest,
            mode=mode,
            dtype=np.double,
            shape=(
                num_sequences,
                num_samples,
                num_features)
        )

        # modalities_assoc
        filename = \
            'modalities_assoc-' + \
            config.VERSION + '.' + \
            config.REVISION + \
            '.mmap'
        dest = os.path.join(config.EXPERIMENTSFOLDER, filename)
        # dest = os.path.join('generated', filename)

        if create:
            modalities_assoc = dest  # haha connard, tu as osé le faire
        else:
            mode = 'rb'
            with open(dest, mode) as f:
                modalities_assoc = pickle.load(f)

        return mmap, modalities_assoc

    def _get_shape(self, X):

        # determine shape of learning dataset
        height = 0  # number of sequences <===== TBD in the following
        width = self.seq_len  # config.FRAMESIZE
        depth = DataReader.num_channels  # number of channels

        for _file in DataReader.trainfiles.values():
            height += int((X[_file].shape[0] - self.seq_len) / self.step + 1)

        return height, width, depth

    def _build_learning_dataset(self, X):
        """
        Synopsis
         construct the learning dataset which will be stored as a memory
         mappeed object in order to leave program's heap alone ...

        Returns
        """
        # determine shape of learning dataset
        shape = self._get_shape(X)

        # allocate memory for learning dataset
        lds, \
            modalities_assoc_filename = \
            self._load_mmap_file(shape, create=True)
        print('lds.shape = %s' % (lds.shape,))

        modalities_assoc = {
            m: []
            for m in DataReader.channels.values()
        }

        try:
            # fill learning dataset with data
            for _file in DataReader.trainfiles.values():
                data = X[_file]
                index = 0
                # discard = data.shape[0] % self.seq_len
                for num_chan, chan_name in DataReader.channels.items():
                    segments = extract_patches(
                        data[:, num_chan],
                        patch_shape=self.seq_len,
                        extraction_step=self.step)
                    lds[:, :, num_chan] = segments
                    print('segments.height = {}'.format(segments.shape[0]))

                    modalities_assoc[chan_name].append(index)
                    index += 1

            # persist offsets_assoc
            print(modalities_assoc)
            with open(modalities_assoc_filename, 'wb+') as f:
                pickle.dump(modalities_assoc, f, pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            print('ERROR: %s' % e)
            print('Failed to build learning dataset')
            print('Removing %s and %s ...' % (lds, modalities_assoc_filename))
            os.remove(lds)
            os.remove(modalities_assoc_filename)
            sys.exit(1)

        return (lds, modalities_assoc)


if __name__ == '__main__':
    # Build data reader and get training data
    dr = DataReader()
    X = dr.data
    p = Segmentation(X)

    X = p.X
    modalities_assoc = p.modalities_assoc
