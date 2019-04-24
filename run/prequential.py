from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import UnsupervisedStream

from models.ae import AE
from models.model import Model
from monitoring.monitoring import BaseMonitor
from data.dataset import Dataset
from config import Configuration


if __name__ == '__main__':

    config = Configuration()
    config.parse_commandline()
    config.new_experiment()

    X = Dataset()[config.CHANNEL]  # already segmented and overlapped!
    stream = UnsupervisedStream(X)
    stream.prepare_for_use()

    # ae = self.clf  (issue #5: Make these tests as classes)
    ae = Model(
        AE(
            # architecture hyperparameters
            n_input=1,
            n_hidden=config.n_hidden,
            n_output=1,
            n_layers=config.n_layers,
        )
    )

    monitor = BaseMonitor(
        ae,
        # learning hyperparameters
        # temperature,
        # learning_rate,
        # weight_decay
    )

    evaluator = EvaluatePrequential(
        n_wait=50,
        pretrain_size=2000,
        batch_size=1,
        metrics=[
            'true_vs_predicted',
            'mean_square_error',
            'mean_absolute_error'
        ],
        output_file=config.EXPERIMENT_PERSISTENCE + '/prequential.npy',
        show_plot=True
    )

    evaluator.evaluate(stream=stream, model=monitor)
