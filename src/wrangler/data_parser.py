###############################################
# Save and load simulation data from/to h5 form
# Developer: Bicheng Yan
# 1/9/2022
# DSFT, KAUST
###############################################


import h5py
import numpy as np
from pathlib import Path


def save_to_h5(data, h5_all_data_path):
    """Save all the data into h5 format."""

    print(f'Save all data into h5 format.')

    # Write data as h5 format
    h5_keys = []
    with h5py.File(h5_all_data_path, 'w') as hf:
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                hf.create_dataset(key, data=val, compression="gzip",compression_opts=9)
                h5_keys.append(key)

    return


def load_from_h5(h5_all_data_path):
    """Load all data from h5."""

    print(f'Load simulation data from h5 format.')

    # Init h5_keys
    h5_keys = ['realization', 'perm_3d', 'poil_3d', 'soil_3d', 'timeSteps']

    # If h5 file already save all the numpy array
    if Path(h5_all_data_path).exists():

        data = {}

        # Get all the numpy array first
        with h5py.File(h5_all_data_path, 'r') as hf:
            for key in h5_keys:
                if key in hf.keys():
                    val = hf.get(name=key)[:]
                    data.update({key: val})
                    print(f'Complete loading {key}.')

        # - dimension parameters
        n_samples, n_time_steps, n_grid_x, n_grid_y, n_grid_z = data['poil_3d'].shape
        data.update({
            # - dimension parameters
            'n_samples': n_samples,
            'n_time_steps': n_time_steps,
            'n_grid_x': n_grid_x,
            'n_grid_y': n_grid_y,
            'n_grid_z': n_grid_z,
        })

    return data

def load_from_h5_general(h5_all_data_path):
    """Load all data from h5."""

    print(f'Load simulation data from h5 format.')

    # If h5 file already save all the numpy array
    if Path(h5_all_data_path).exists():

        data = {}

        # Get all the numpy array first
        with h5py.File(h5_all_data_path, 'r') as hf:
            for key in hf.keys():
                val = hf.get(name=key)[:]
                data.update({key: val})
                print(f'Complete loading {key}.')

    return data


