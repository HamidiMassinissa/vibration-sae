# -*- coding: utf-8 -*-
import os
import datetime
from argparse import ArgumentParser


class Configuration:

    # experiment version
    MAJOR_VERSION = '0'
    MINOR_VERSION = '1'
    VERSION = MAJOR_VERSION + '.' + MINOR_VERSION
    REVISION = '0'
    MINOR_REVISION = '0'

    # paths
    PREFIX = ''
    DATAFOLDER = PREFIX + 'data'
    EXPERIMENTSFOLDER = PREFIX + '_experiments'
    EXPERIMENT_PERSISTENCE = EXPERIMENTSFOLDER
    BO_RUN_PERSISTENCE = EXPERIMENT_PERSISTENCE
    # RECONSTRUCTIONSFOLDER = PREFIX + 'experiments/reconstructions'
    # RECONSTRUCTIONS_TO_DISPLAY_FILENAME = PREFIX + 'reconstructions-to-display.bin'

    SEED = 1
    N_CALLS = 60
    N_JOBS_bayes = 1

    FINGERPRINT_SIZE = 12
    CHANNEL = 'acc2__'
    TIMESTAMP_CHANNEL = 'time__'
    DEBUG_CHANNEL = 'acc2__'
    DEBUG = True
    LOG_INTERVAL = 10
    MAX_TRAINING_EPOCHS = 2
    CUDA = False
    CV_N_SPLITS = 10

    # cv_iter = 0
    batch_idx = 0
    epoch_idx = 0

    # hyperparameters for debug, NB. hp are lowercase
    framesize = 50
    overlap = 0.9
    batch_size = 10
    learning_rate = 1e-2
    temperature = 0.5
    weight_decay = 1
    n_layers = 1
    n_hidden = 20
    encoder_num_hidden_units_1 = 100
    encoder_num_hidden_units_2 = 100
    decoder_num_hidden_units_1 = 100
    decoder_num_hidden_units_2 = 100
    input_keep_probability = 5e-1
    output_keep_probability = 5e-1
    state_keep_probability = 5e-1
    independent_batches = True
    allow_hidden_to_flow = False
    max_norm = 0.25
    sparsity = 0.05
    sparsity_penalty = 0.5

    @classmethod
    def parse_commandline(self, is_testing=False):
        """
        Synopsis
         Alter class attributes defined above with command line arguments
        """
        parser = ArgumentParser(description='')
        parser.add_argument(  # absolute path of the current directory
            '--run',
            metavar='run',
            required=True
        )
        parser.add_argument(  # absolute path of the current directory
            '--prefix',
            metavar='prefix',
            required=False
        )
        parser.add_argument(
            '--revision',
            metavar='revision',
            default='0',
            required=False
        )
        parser.add_argument(
            '--minor-revision',
            metavar='minor_revision',
            default='0',
            required=False
        )
        parser.add_argument(
            '--channel',
            metavar='channel',
            default='acc2__',
            required=False
        )
        parser.add_argument(
            '--independent-batches',
            metavar='independent_batches',
            required=False
        )
        parser.add_argument(
            '--allow-hidden-to-flow',
            metavar='allow_hidden_to_flow',
            required=False
        )
        args = parser.parse_args()
        self.RUN = args.run
        self.PREFIX = args.prefix
        self.REVISION = args.revision
        self.MINOR_REVISION = args.minor_revision
        self.CHANNEL = args.channel
        self.independent_batches = args.independent_batches
        self.allow_hidden_to_flow = args.allow_hidden_to_flow

    @classmethod
    def __str__(cls):
        return ', '.join(
            '{}: {}\n'.format(k, v)
            for (k, v) in cls.__dict__.items()  # if k.startswith('_')
        )

    @classmethod
    def new_experiment(self):
        self.EXPERIMENT_PERSISTENCE = os.path.join(
            self.EXPERIMENT_PERSISTENCE, '{}'.format(datetime.datetime.now()))
        self.BO_RUN_PERSISTENCE = self.EXPERIMENT_PERSISTENCE
        assert not os.path.exists(self.EXPERIMENT_PERSISTENCE)
        os.makedirs(self.EXPERIMENT_PERSISTENCE)

    @classmethod
    def new_BO_run(self):
        self.BO_RUN_PERSISTENCE = os.path.join(
            self.EXPERIMENT_PERSISTENCE, '{}'.format(datetime.datetime.now()))
        assert not os.path.exists(self.BO_RUN_PERSISTENCE)
        os.makedirs(self.BO_RUN_PERSISTENCE)
