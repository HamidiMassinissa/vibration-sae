import torch
import numpy as np

from .utils import save
# from .tSNE import plot_tSNE, record_tSNE_gradient_descent
from config import Configuration as config


def train_epoch(model, dataset, criterion, optimizer):
    """train_epoch

    Args:
        independent_batches: When your batches of data are independent short
            sequences, for example sentences of text, then you should
            reinitialise the hidden state before each batch. But if your data is
            made up of really long sequences like stock price data, and you cut
            it up into batches making sure that each batch follows on from the
            previous batch, then in that case you wouldn’t reinitialise the
            hidden state before each batch.
    """
    model.train()
    loss_values = []
    encodings = []
    # print('################ {}'.format(dataset.shape))
    # for i, (x, _) in enumerate(dataset):  # dataset is supposed to yield batches
    for i, x in enumerate(dataset):  # dataset is supposed to yield batches
        # print('############### {}'.format(x))
        x = torch.from_numpy(x).float().view(1, -1, 1)
        # x = model.bn(x)
        # if not config.allow_hidden_to_flow:
        initial_hidden = model.init_hidden(x.shape[0])
        prediction, encoding = model(x, initial_hidden, temperature=0.5, return_encoding=True)
        # print('I: [train_epoch] prediction.shape = {}'.format(prediction.shape))
        # print('I: [train_epoch] prediction = %s' % np.array2string(prediction.detach().numpy(), threshold=np.inf).replace('\n', ''))
        # print('I: [train_epoch] y.shape = {}'.format(x.shape))
        # loss = criterion(prediction, y)
        loss = criterion(prediction, x)
        loss_values.append(loss.detach().cpu().numpy())
        # if config.independent_batches:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        # utils.print_model_stats(model)
        # utils.plot_prediction(x, prediction, encoding, 'train')
        # utils.plot_encoding(encoding)
        encodings.append(encoding.detach().numpy())
        config.batch_idx = i
        if i % config.LOG_INTERVAL == 0:
            print('I: [train_epoch] batch-{:3} Loss {}'.format(i, loss))

    if not config.independent_batches:
        optimizer.zero_grad()

    save('train_loss_values.npy', loss_values)
    save('encodings.npy', encodings)
    # plot_tSNE(np.concatenate(encodings, axis=0))
    # tSNE.record_tSNE_gradient_descent(np.concatenate(encodings, axis=0))
    return np.array(loss_values)


def test(model, dataset, criterion):
    model.eval()
    loss_values = []
    with torch.no_grad():  # switching off gradients makes things faster
        for i, (x, _) in enumerate(dataset):
            # x = model.bn(x)
            initial_hidden = model.init_hidden(x.shape[0])
            prediction, encoding = model(
                x, initial_hidden, temperature=0., return_encoding=True)
            loss = criterion(prediction, x)
            loss_values.append(loss.detach().cpu().numpy())
            # print('I: [test] batch-{:3} Loss {}'.format(i, loss))
            # utils.plot_prediction(x, prediction, encoding, 'test')
            config.batch_idx = i
    save('test_loss_values.npy', loss_values)
    return np.array(loss_values)


def train(model, dataset, learning_rate=0.1, weight_decay=0.1, n_epochs=1,
          test_dataset=None):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    test_losses = []
    for i in range(n_epochs):
        print('I: [train]@epoch {:3}'.format(i))
        config.epoch_idx = i
        train_loss = train_epoch(model, dataset, criterion, optimizer)
        train_losses.append(train_loss)
        if test_dataset is not None:
            test_loss = test(model, test_dataset, criterion)
            test_losses.append(test_loss)
    return np.array(train_losses), np.array(test_losses)


# def one_step_ahead_prediction(model, x): prediction is performed as in the
# training phase!
def predict(model, x):
    model.eval()
    with torch.no_grad():
        initial_hidden = model.init_hidden(x.shape[0])
        x = torch.from_numpy(x).float().view(1, -1, 1)
        prediction, encoding = model(
            x, initial_hidden, temperature=0., return_encoding=True)

    # import pdb
    # pdb.set_trace()
    return prediction.numpy()[0][-1]


def reset_weights(m):
    if isinstance(m, torch.nn.LSTM) or \
            isinstance(m, torch.nn.LSTMCell) or \
            isinstance(m, torch.nn.Linear):
        m.reset_parameters()
