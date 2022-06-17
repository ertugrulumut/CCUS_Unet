import torch
import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(0)
torch.manual_seed(0)


def count_parameters(model):
    """Count the total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def RMSELoss(yhat,y):
    if isinstance(yhat, np.ndarray):
        yhat_, y_ = yhat.copy(), y.copy()
        yhat_[yhat_ == 0.0] = np.nan  # Remove zeros in the yhat_
        y_[y_ == 0.0] = np.nan        # Remove zeros in the y_
        return np.sqrt(np.nanmean((yhat_-y_)**2))
    else:
        return torch.sqrt(torch.mean((yhat-y)**2))


def get_device(d='cpu'):
    """Set up CPU (default) or GPU environment for pytorch."""

    if torch.cuda.is_available() and d.upper() == 'GPU':
        return torch.device(f'cuda:0')
    else:
        return torch.device('cpu')


def get_datetime():
    """Return the current computer date and time in string."""
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def get_temporal_loss(data_true, data_pred):
    """Get the loss of pressure and saturation at each timestep."""

    # Error collector
    losses = []
    time_steps = data_true['timeSteps']

    # Loop each timesteps: get temporal error
    for j, jtstep in enumerate(time_steps):

        # Get total loss of all the cases
        if 'pres_3d' in data_pred:
            pres_3d_bulk_loss = RMSELoss(data_true['pres_3d'][:, j, :, :, :],
                                         data_pred['pres_3d'][:, j, :, :, :])
        else:
            pres_3d_bulk_loss = np.nan
        if 'sG_3d' in data_pred:
            sG_3d_bulk_loss = RMSELoss(data_true['sG_3d'][:, j, :, :, :],
                                         data_pred['sG_3d'][:, j, :, :, :])
        else:
            sG_3d_bulk_loss = np.nan

        # Save them
        losses.append([jtstep, pres_3d_bulk_loss, sG_3d_bulk_loss])

    return pd.DataFrame(data=losses, columns=['days', 'pres_loss', 'sG_loss'])


def get_loss(data_true, data_pred):
    """Get loss of all the predictions"""

    # Get total loss of all the cases
    if 'pres_3d' in data_pred:
        pres_3d_loss = RMSELoss(data_true['pres_3d'][:, :, :, :, :].astype(float),
                                data_pred['pres_3d'][:, :, :, :, :].astype(float))
    else:
        pres_3d_loss = np.nan
    if 'sG_3d' in data_pred:
        sG_3d_loss = RMSELoss(data_true['sG_3d'][:, :, :, :, :].astype(float),
                                data_pred['sG_3d'][:, :, :, :, :].astype(float))
    else:
        sG_3d_loss = np.nan

    return {
        'pres_3d_loss': pres_3d_loss,
        'sG_3d_loss': sG_3d_loss,
    }


def train_val_test_split(nsamples, test_size=0.1, val_size=0.1):
    """Split data based on test and val size."""

    # Shuffle the samples first
    samples_shuffled = np.random.permutation(nsamples)

    # Get number of cases
    num_test_cases = int(nsamples * test_size)
    num_validation_cases = int(nsamples * val_size)
    test_cases = samples_shuffled[:num_test_cases].tolist()
    validation_cases = samples_shuffled[num_test_cases:num_test_cases+num_validation_cases].tolist()
    train_cases = list(set(samples_shuffled.tolist()).difference(test_cases).difference(validation_cases))

    test_cases.sort()
    train_cases.sort()
    validation_cases.sort()

    return train_cases, test_cases, validation_cases

