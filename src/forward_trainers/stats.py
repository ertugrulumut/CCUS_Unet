###############################################
# Preprocessing scaler for data
# Developer: Bicheng Yan
# 9/1/2021
# DSFT, KAUST
###############################################


import json
import numpy as np


class TorchMinMaxScaler():
    """A minmax scaler allow for both torch tensor and numpy array operation, allow tensor
    and numpy array operation both."""

    def __init__(self):

        self.min = 0.0
        self.max = 0.0
        self.range = 0.0
        self.fitted = False
        self.type = 'MinMax'

    def fit(self, array):
        """Get the min, max, range of values in array."""

        # Just get the min, max, and rang of array
        self.min = array.min()
        self.max = array.max()

        # Check if it is a constant
        if self.min == self.max:
            self.min = 0.0

        self.range = self.max - self.min
        self.fitted = True

    def fit_transform(self, array):
        """Scale array, here."""

        if self.fitted:
            scaled_array = (array - self.min) / self.range
        else:
            self.fit(array)
            scaled_array = (array - self.min) / self.range

        return scaled_array

    def inverse_transform(self, scaled_array):
        """Inversely scale scaled_array."""

        if not self.fitted:
            raise Exception('ERROR: the scaler is not fitted.')
        else:
            array = scaled_array * self.range + self.min

        return array


def create_scalers(data):
    """
    Create scalers for different channels of data generated from cmg parser,
    :param data: data generated from cmg parser,
    :param scaler_type: either 'MinMax' (default) or 'Standard',
    :return: dictionary of scalers
    """

    scaler_dict = dict()
    new_data = dict()

    # Loop through each key in data
    for key, val in data.items():

        # Only scale numpy array
        if not isinstance(val, np.ndarray):
            new_data.update({
                key: val,
            })
            continue

        # Create a scaler
        scaler = TorchMinMaxScaler()

        # Update scaler_dict and new_data
        new_val = scaler.fit_transform(val)
        new_data.update({
            key: new_val,
        })

        scaler_dict.update({
            key: scaler
        })

    return scaler_dict, new_data


def load_scalers(scaler_meta):

    scaler_dict = {}

    for k, v in scaler_meta.items():
        if v['type'] == 'MinMax':
            scaler = TorchMinMaxScaler()
            scaler.min = v['min']
            scaler.max = v['max']
            scaler.range = v['range']
            scaler.fitted = True
            scaler_dict.update({k: scaler})

    return scaler_dict


def save_scaler_meta(scaler_dict, file_path):

    # Init scaler_meta
    scaler_meta = {}

    for k, v in scaler_dict.items():
        if isinstance(v, TorchMinMaxScaler):
            vtype = 'MinMax'
            scaler_meta.update({k: {
                'type': vtype,
                'min': float(v.min),
                'max': float(v.max),
                'range': float(v.range),
            }})
        else:
            vtype = 'Gaussian'
            scaler_meta.update({k: {
                'type': vtype,
                'mean': float('v.mean'),
                'std': float(v.std),
            }})

    # Save into file_path
    with open(file_path, "w") as outfile:
        json.dump(scaler_meta, outfile, indent=4)


