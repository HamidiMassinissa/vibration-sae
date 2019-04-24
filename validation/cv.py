import numpy as np
import torch.utils.data  # In particular, DataLoader yields batches
from sklearn.model_selection import KFold

from models.train import train, reset_weights
from models.zipdataset import ZipDataset
from validation.utils import save_txt


def cv(model,
       data,
       target,
       temperature,
       weight_decay,
       learning_rate,
       sparsity,
       sparsity_penalty,
       n_epochs,
       n_splits,
       seed,
       batch_size,
       shuffle):

    cv = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed)
    train_losses = []
    test_losses = []
    for cv_iter, (train_index, test_index) in enumerate(cv.split(data, target)):
        print('I: [cross_val_train] cv_iter = {:2}'.format(cv_iter))
        model.apply(reset_weights)  # reset model's weights, recursively

        train_data = data[train_index]
        train_target = target[train_index]
        train_dataloader = torch.utils.data.DataLoader(
            ZipDataset(train_data[:, :-1], train_target[:, 1:]),
            batch_size=batch_size,
            shuffle=shuffle)
        test_data = data[test_index]
        test_target = target[test_index]
        test_dataloader = torch.utils.data.DataLoader(
            ZipDataset(test_data[:, :-1], test_target[:, 1:]),
            batch_size=batch_size,
            shuffle=shuffle)

        cv_train_loss, cv_test_loss = train(
            model,
            train_dataloader,
            temperature=temperature,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            sparsity=sparsity,
            sparsity_penalty=sparsity_penalty,
            n_epochs=n_epochs,
            test_dataset=test_dataloader)
        # plot_loss(cv_train_loss, 'cv_train_loss:cv_iter_{}'.format(cv_iter))
        # save(cv_train_loss, 'cv_train_loss:cv_iter_{}'.format(cv_iter))

        train_losses.append(cv_train_loss)
        test_losses.append(cv_test_loss)

    mean = np.mean(test_losses[-1])
    var = np.var(test_losses[-1])
    save_txt(train_losses, 'train_loss')
    save_txt(test_losses, 'test_loss')

    print('I: [cross_val_train] mean of test losses = {:2}'.format(mean))
    print('I: [cross_val_train] variance of test losses = {:2}'.format(var))

    return mean
