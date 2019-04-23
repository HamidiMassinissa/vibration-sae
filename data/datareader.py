import sys
import fire
import os
import subprocess
import numpy as np
import pandas as pd
from .utils import float_of_strDateTime

from config import Configuration as config


# DataReader : file list -> data
class DataReader(object):
    def __init__(self):

        # before starting anything, check if the right folder where we will
        # store data exists, otherwise create it
        if not os.path.exists(config.EXPERIMENTSFOLDER):
            os.makedirs(config.EXPERIMENTSFOLDER)

        self._data = self._load_data()

    @property
    def data(self):
        return self._data

    # columns to be loaded
    columns = {
        'timestamp':    'DateTime',
        'acc1':         'DEP102J:102AX1AD1.PNT',
        'acc2':         'DEP102J:102AX4AD1.PNT',
        'speed':        'SP102J:102SP01.PNT',
        'vib1_h':       'VIB102J:102VIB01HD.PNT',
        'vib1_v':       'VIB102J:102VIB01VD.PNT',
        'vib2_h':       'VIB102J:102VIB02HD.PNT',
        'vib2_v':       'VIB102J:102VIB02VD.PNT',
        'vib3_h':       'VIB102J:102VIB03HD.PNT',
        'vib3_v':       'VIB102J:102VIB03VD.PNT',
        'vib4_h':       'VIB102J:102VIB04HD.PNT',
        'vib4_v':       'VIB102J:102VIB04VD.PNT'
    }

    data_generation_model = {
        0: ['acc1'],
        1: ['vib1_h', 'vib1_v'],
        2: ['vib2_h', 'vib2_v'],
        3: ['vib3_h', 'vib3_v'],
        4: ['vib4_h', 'vib4_v'],
        5: ['acc2'],
        6: ['speed']
    }

    channels = {
        0:  'time__',
        1:  'acc1__',
        2:  'acc2__',
        3:  'speed_',
        4:  'vib1_h',
        5:  'vib1_v',
        6:  'vib2_h',
        7:  'vib2_v',
        8:  'vib3_h',
        9:  'vib3_v',
        10: 'vib4_h',
        11: 'vib4_v'
    }

    trainfiles = {
        1: 'Book1-attempt1.csv'
    }

    testfiles = {
    }

    num_columns = len(columns)  # 12
    num_channels = len(channels)  # 12

    def _load_data(self):
        """
        Synopsis

         Returns
        """
        data = {}

        for _file in self.trainfiles.values():
            data[_file] = {}

            src = os.path.join(
                config.DATAFOLDER,
                _file)

            pipe = subprocess.Popen(
                "wc -l < " + src,
                shell=True,
                stdout=subprocess.PIPE).stdout
            num_rows = int(pipe.read()) - 1  # discard header row

            dest = os.path.join(
                config.EXPERIMENTSFOLDER,
                _file + '.mmap')

            data[_file] = self._mmap_file(
                src,
                dest,
                dtype=np.double,
                shape=(int(num_rows), self.num_columns))

        return data

    def _mmap_file(self, src, dest, dtype, shape):
        if os.path.exists(dest):
            # just load mmap file contents
            print('%s exists, loading ...' % dest)
            mmap = np.memmap(
                dest,
                mode='r+',
                dtype=dtype,
                shape=shape)

            # debug
            print('[_mmap_file] %s' % dest)
            print(mmap)

            return mmap
        else:
            try:
                # build mmap file from scratch
                print('Building from scratch %s ...' % dest)
                print(shape)
                mmap = np.memmap(
                    dest,
                    mode='w+',
                    dtype=dtype,
                    shape=shape)

                chunksize = 5000
                offset = 0
                for chunk in pd.read_csv(src, delimiter=',',
                                         chunksize=chunksize, header=0):
                    chunk['DateTime'] = chunk['DateTime'].map(float_of_strDateTime)
                    print('offset %d' % offset)
                    print('DateTime:%s | column1:%s |' % (chunk.values[0, 0], chunk.values[0, 1]))
                    mmap[offset:offset+chunk.shape[0]] = chunk.values
                    offset += chunk.shape[0]

                # debug
                print('[_mmap_file] %s' % dest)
                print(mmap)

                return mmap
            except Exception as e:
                print('ERROR: %s' % e)
                print('Failed to build file from scratch')
                print('Removing %s ...' % dest)
                os.remove(dest)
                sys.exit(1)


if __name__ == '__main__':
    # Python Fire is a library for automatically generating command line
    # interfaces (CLIs) from absolutely any Python object.
    fire.Fire(DataReader)
