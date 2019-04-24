import torch
import numpy as np
from scipy.stats import entropy

from .utils import save
# from .tSNE import plot_tSNE, record_tSNE_gradient_descent
from config import Configuration as config


def train_epoch(model, dataset, criterion, optimizer):
    """train_epoch
    """
    model.train()
    loss_values = []
    encodings = []
    for i, x in enumerate(dataset):  # dataset is supposed to yield batches
        x, y = x
        # x = torch.from_numpy(x[0]).float().view(1, -1, 1)
        initial_hidden = model.init_hidden(x.shape[0])
        prediction, encoding = model(x, initial_hidden, temperature=0.5, return_encoding=True)
        loss = criterion(prediction, x)
        # rho = torch.FloatTensor([RHO for _ in range(N_HIDDEN)]).unsqueeze(0)
        # rho_hat = torch.sum(encoded, dim=0, keepdim=True)
        # sparsity_penalty = BETA * entropy(rho, rho_hat)
        # loss = MSE_loss + sparsity_penalty
        loss_values.append(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        encodings.append(encoding.detach().numpy())
        config.batch_idx = i
        if i % config.LOG_INTERVAL == 0:
            print('I: [train_epoch] batch-{:3} Loss {}'.format(i, loss))

    if not config.independent_batches:
        optimizer.zero_grad()

    save('train_loss_values.npy', loss_values)
    save('encodings.npy', encodings)
    return np.array(loss_values)


def test(model, dataset, criterion):
    model.eval()
    loss_values = []
    with torch.no_grad():  # switching off gradients makes things faster
        for i, (x, _) in enumerate(dataset):
            initial_hidden = model.init_hidden(x.shape[0])
            prediction, encoding = model(
                x, initial_hidden, temperature=0., return_encoding=True)
            loss = criterion(prediction, x)
            loss_values.append(loss.detach().cpu().numpy())
            config.batch_idx = i
    save('test_loss_values.npy', loss_values)
    return np.array(loss_values)


def train(model,
          dataset,
          temperature=0.2,
          weight_decay=0.1,
          learning_rate=0.1,
          sparsity=0.05,
          sparsity_penalty=0.5,
          n_epochs=1,
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
