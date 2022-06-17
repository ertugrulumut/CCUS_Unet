###############################################
# Reservoir simulation basic input data
# Developer: Bicheng Yan
# 1/10/2022
# DSFT, KAUST
###############################################


import numpy as np


def get_well_2d_map(n_grid_x, n_grid_y, well_grid_index):
    """Get 2D well locations in 2D reservoir domain."""

    well_2d = np.zeros(shape=(n_grid_x, n_grid_y), dtype=float)
    for (i, j) in well_grid_index:
        well_2d[i, j] = 1.0

    return well_2d

N_GRID_X = 51
N_GRID_Y = 51
N_GRID_Z = 1

INJ_GRIDS = np.array([
    [5, 10],
    [31, 15],
    [45, 28],
    [14, 46],
    [34, 43],
]) - 1

PROD_GRIDS = np.array([
    [28, 26],
]) - 1


PROD_2D_MAP = get_well_2d_map(N_GRID_X, N_GRID_Y, PROD_GRIDS)

INJ_2D_MAP = get_well_2d_map(N_GRID_X, N_GRID_Y, INJ_GRIDS)

