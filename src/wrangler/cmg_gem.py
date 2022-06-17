###############################################################################################
# Parse CMG GEM Output files into dictionaries of 3D numpy arrays vs time
# Developer: Bicheng Yan
# 3/29/2022
# DSFT, KAUST
###############################################################################################

import json
import glob
import time
import numpy as np
from pathlib import Path

from wrangler.data_parser import save_to_h5, load_from_h5_general

def peaceman_well_index(dx, dy, dz, permx, permy):
    """
    Calculate well index based on peaceman equation,
    :param dx: grid size in x direction, ft
    :param dy: grid size in y direction, ft
    :param dz: grid size in z direction, ft
    :param perm: permeability of the
    :param well_dir: 'x', 'y' or 'z' direction,
    :return: well index.
    """

    r_w = 0.2    # ft
    r_o = 0.28 * np.sqrt((dx**2) * np.sqrt(permy/permx) + (dy**2) * np.sqrt(permx / permy)) / \
          ((permy/permx)**0.25 + (permx/permy)**0.25)

    WI = 2.0 * np.pi * dz * np.sqrt(permx*permy) / (np.log(r_o/r_w))

    return WI


class GEM_File:

    grid_search_keywords = ['*GRID','*CART']
    time_search_keywords = ['Time','=']# ['Time','=','hr']
    coord_search_keywords = ['*DI', '*DJ', '*DK']

    def __init__(self, file_name, verb=0):

        self.file_name = file_name
        self.verb = verb
        self.input_list = self.read_file(self.file_name)
        self.current = 0 # pointer to current line number in file
        self.nx, self.ny, self.nz = self.get_grid_dim(self.grid_search_keywords)
        self.dx, self.dy, self.dz = None, None, None

    # read file into list
    def read_file(self, file_name):
        input_list = []
        with open(file_name) as f:
            input_list = f.readlines()
        return input_list

    # read grid dimensions
    def get_grid_dim(self, search_strings):
        for m,line in enumerate(self.input_list[self.current:]):
            if all(elem in line for elem in search_strings):
                line_list = line.split()
                self.current = self.current + m
                return int(line_list[-3]), int(line_list[-2]), int(line_list[-1])
        self.current = -1
        return -1, -1, -1

    # read grid size
    def get_grid_size(self):
        """Get the grid volume and grid coordinates of each cell."""

        self.current = 0
        search_strings = self.coord_search_keywords
        dx = np.zeros((self.nx, ))
        dy = np.zeros((self.ny, ))
        dz = np.zeros((self.nz, ))
        for m,line in enumerate(self.input_list[self.current:]):
            if any(elem in line for elem in search_strings):
                line_list = line.split()
                # Check *CON in list_line
                if ('*CON' in line_list) and ('*DI' in line_list):
                    dx[:] = float(line_list[2])
                if ('*CON' in line_list) and ('*DJ' in line_list):
                    dy[:] = float(line_list[2])
                if ('*CON' in line_list) and ('*DK' in line_list):
                    dz[:] = float(line_list[2])
                self.current = self.current + m

        # Save to dx, dy, dz
        self.dx, self.dy, self.dz = dx, dy, dz

        # Get grid volume
        grid_volume = np.zeros((self.nx, self.ny, self.nz))
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    grid_volume[i, j, k] = dx[i] * dy[j] * dz[k]

        # Get grid coordinates
        xcoord, ycoord, zcoord = np.zeros((self.nx, )), np.zeros((self.ny, )), np.zeros((self.nz, ))
        xcoord[0], ycoord[0], zcoord[0] = dx[0]*0.5, dy[0]*0.5, dz[0]*0.5
        for i in range(1, self.nx, 1):
            xcoord[i] = xcoord[i-1] + (dx[i-1] + dx[i]) * 0.5
        for j in range(1, self.ny, 1):
            ycoord[j] = ycoord[j-1] + (dy[j-1] + dy[j]) * 0.5
        for k in range(1, self.nz, 1):
            zcoord[k] = zcoord[k-1] + (dz[k-1] + dz[k]) * 0.5

        grid_xcoord = np.zeros((self.nx, self.ny, self.nz))
        grid_ycoord = np.zeros((self.nx, self.ny, self.nz))
        grid_zcoord = np.zeros((self.nx, self.ny, self.nz))
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    grid_xcoord[i, j, k] = xcoord[i]
                    grid_ycoord[i, j, k] = ycoord[j]
                    grid_zcoord[i, j, k] = zcoord[k]

        return grid_volume.tolist(), grid_xcoord.tolist(), grid_ycoord.tolist(), grid_zcoord.tolist()


    # find next time step
    def get_time(self, search_strings):
        for m,line in enumerate(self.input_list[self.current+1:]):
            if all(elem in line for elem in search_strings):
                line_list = line.split()
                i = line_list.index('=')
                # save line number, return time and units
                self.current = self.current + m + 1
                return float(line_list[i+1]), line_list[i+2]
        self.current = -1 # end of file reached
        return -1.0, 'days'   

    # move file pointer to line containing all search strings in list
    def find_it(self, search_strings):
        if (self.current < 0):
            return
        for m,line in enumerate(self.input_list[self.current+1:]):
            if all(elem in line for elem in search_strings):
                self.current = self.current + m + 1
                return
        return

    # read output variable into 2D numpy array
    def read_variable(self):
        self.current = self.current + 3

        # Loop for all the planes
        var = np.zeros((self.nx, self.ny, self.nz), dtype=np.float)
        for plane_num in range(self.nz):
            line = self.input_list[self.current]
            if all(elem in line for elem in ['All','values','are']):
                # constant
                line_list = line.split()
                var[:, :, plane_num] = np.ones((self.nx,self.ny))*float(line_list[-1])
                self.current = self.current + 2
            elif all(elem in line for elem in ['Plane', 'K', '=']) and not all(elem in line for elem in ['I','=']):
                # Read a line more
                self.current = self.current + 1
                line = self.input_list[self.current]

                # variable
                while True:
                    line_list = line.split()
                    int_list = list(map(int, line_list[2:]))
                    ngrids = len(int_list)

                    # Get the location of each value in the string
                    tmp_current = self.current
                    for j in range(self.ny):
                        tmp_current += 1
                        line = self.input_list[tmp_current]
                        line_list = line.split()
                        float_list = list(map(float, line_list[2:]))
                        if len(float_list) == ngrids:
                            str_ix = []
                            end_ix = []
                            for r in range(ngrids):
                                ix = line.find(str(float_list[r]))
                                str_ix.append(ix)
                                end_ix.append(ix+len(str(float_list[r])))
                                line = line[:ix] + '*' + line[ix+1:] # Add s special string to help pop out float_list[r]

                            # Once str_ix and end_ix recorded, break the j-loop
                            break

                    for j in range(self.ny):
                        self.current = self.current + 1
                        line = self.input_list[self.current]
                        line_list = line.split()
                        float_list = list(map(float, line_list[2:]))
                        # In case null grid, need to match the location of each value based on str_ix and end_ix
                        if len(float_list) != ngrids:
                            float_list = [float(line[str_ix[r]:end_ix[r]]) if line[str_ix[r]] != ' ' else 0.0 for r in range(ngrids)]
                        for m,n in enumerate(int_list):
                            i = n - 1
                            var[i][j][plane_num] = float_list[m]
                    if int_list[-1] == self.ny:
                        self.current += 2
                        break
                    self.current = self.current + 2
                    line = self.input_list[self.current]

            elif all(elem in line for elem in ['I','=']):
                # variable
                var = np.zeros((self.nx,self.ny, 1))
                while True:
                    line_list = line.split()
                    int_list = list(map(int, line_list[2:]))
                    for j in range(self.ny):
                        self.current = self.current + 1
                        line = self.input_list[self.current]
                        line_list = line.split()
                        float_list = list(map(float, line_list[2:]))
                        for m,n in enumerate(int_list):
                            i = n - 1
                            var[i][j][0] = float_list[m]
                    if int_list[-1] == self.ny:
                        break
                    self.current = self.current + 2
                    line = self.input_list[self.current]
        return var

    # get output variable at all time steps
    def get_variable(self, search_strings, static=False):
        self.current = 0
        variables = {}
        current_timestep = -1.0
        while True:
            # look for next time step
            timestep, units = self.get_time(self.time_search_keywords)
            if (timestep == current_timestep):
                continue
            if (self.current == -1.0):
                break
            # get the selected variable values
            self.find_it(search_strings)
            if self.verb > 0:
                print(f'time {timestep} at line {self.current + 1}, search for {search_strings}')
            var = self.read_variable()
            variables.update({timestep : var.tolist()})
            current_timestep = timestep
            if static: break
        return variables

    # get well coordinates
    def get_well_coord(self, well_num, location_strings):
        # find well coordinates
        well_str = str(well_num)
        location_strings.append(well_str)
        self.find_it(location_strings)
        self.current = self.current + 1
        line = self.input_list[self.current]
        line_list = line.split()

        # Split i, j, k index
        num_perfs = 1
        well_ijk = []
        for n in range(3):
            s = line_list[n]
            if ':' in s:
                str_ind = int(s.split(':')[0])
                end_ind = int(s.split(':')[1])
                inds = [*range(str_ind, end_ind+1)]
                num_perfs = len(inds)
            else:
                inds = [int(line_list[n])]
            well_ijk.append(inds)
        # Correct dimension
        well_ijk = np.array([
            inds*num_perfs if len(inds) != num_perfs else inds for inds in well_ijk
        ])

        return well_ijk[0, :], well_ijk[1, :], well_ijk[2, :]

    # get well rate at a particular time
    def get_well_rate(self, well_num, search_strings, subsearch_strings):

        # look for well rate
        self.find_it(search_strings)
        self.find_it(subsearch_strings)
        line = self.input_list[self.current]
        line_list = line.split()[len(subsearch_strings):]

        return float(line_list[well_num*2-1])

    # get well rates as a dictionary of rate vs time
    def get_well_rates(self, well_num, search_strings, subsearch_strings):
        self.current = 0
        values = {}
        # look for time zero
        timestep, units = self.get_time(self.time_search_keywords)
        # no injection at time zero
        values.update({timestep : 0.0})
        current_timestep = timestep
        while True:
            # look for next time step
            timestep, units = self.get_time(self.time_search_keywords)
            if (timestep == current_timestep):
                continue
            if (self.current == -1.0):
                break
                # return values # end of file

            # Search 'G E M   F I E L D  S U M M A R' first
            self.find_it(search_strings='G E M   F I E L D  S U M M A R')
            tmp_current = self.current
            # Try to find a rate first, if the line number is too far away from tmp_current, this timestep should have 0 rate
            # - caveat: sometimes prod/inj rates are not available under 'GEM FIELD SUMMARY'
            try:
                rate = self.get_well_rate(well_num, search_strings, subsearch_strings)
            except:
                rate = 0.0
            if self.current - tmp_current > 300:
                rate = 0.0
                self.current = tmp_current

            if self.verb > 0:
                print(f'time {timestep} at line {self.current + 1}, search for {search_strings} && {subsearch_strings} = {rate}.')
            values.update({timestep : rate})
            current_timestep = timestep
        return values


    # get well rates and well index as a dictionary of 3D numpy arrays
    def get_well_maps(self, well_num, location_strings, search_strings, subsearch_strings, perm_field):
        self.current = 0
        variables = {}
        well_i, well_j, well_k = self.get_well_coord(well_num, location_strings)
        num_perfs = len(well_i)

        # Get well index
        WI = np.zeros((self.nx, self.ny, self.nz))
        for i in well_i:
            for j in well_j:
                for k in well_k:
                    WI[i-1, j-1, k-1] = peaceman_well_index(self.dx[i-1], self.dy[j-1], self.dz[k-1], perm_field[i-1, j-1, k-1], perm_field[i-1, j-1, k-1])
        WI_ratio = WI / np.sum(WI[:])

        # look for time zero
        timestep, units = self.get_time(self.time_search_keywords)
        # no injection at time zero
        var = np.zeros((self.nx,self.ny, self.nz))
        variables.update({timestep : var.tolist()})
        # find values at each time step
        current_timestep = timestep
        while True:
            # look for next time step
            timestep, units = self.get_time(self.time_search_keywords)
            if (timestep == current_timestep):
                continue
            if (self.current == -1.0):
                break
                # return variables # end of file
            # put value at well location on grid
            var = np.zeros((self.nx,self.ny,self.nz))

            #rate = self.get_well_rate(well_num, search_strings, subsearch_strings)
            # Search 'G E M   F I E L D  S U M M A R' first
            self.find_it(search_strings='G E M   F I E L D  S U M M A R')
            tmp_current = self.current
            # Try to find a rate first, if the line number is too far away from tmp_current, this timestep should have 0 rate
            # - caveat: sometimes prod/inj rates are not available under 'GEM FIELD SUMMARY'
            rate = self.get_well_rate(well_num, search_strings, subsearch_strings)
            if self.current - tmp_current > 300:
                rate = 0.0
                self.current = tmp_current

            if self.verb > 0:
                print(f'time {timestep} at line {self.current + 1}, search for {search_strings} && {subsearch_strings} = {rate}.')
            for n in range(num_perfs):
                i, j, k = well_i[n]-1, well_j[n]-1, well_k[n]-1
                # Arithmatic split to each perforation <== can improve later based on well index magnitutude
                var[i, j, k] = rate * WI_ratio[i, j, k]
            variables.update({timestep : var.tolist()})
            current_timestep = timestep
        return variables, WI.tolist()


def cmg_parser(data_path, drop_timesteps=[]):
    """
    Load injection/production rates (maps), pressure, gas saturation, perm, porosity field from CMG files.
    :param data_path: file path where the CMG output files are saved,
    :param drop_timesteps: time steps for some samples not available, choose to drop it for dimension consistency.

    :return n_samples: number of samples,
    :return n_time_steps: number of timesteps,
    :return n_grid_x: number of grids in the x-dir,
    :return n_grid_y: number of grids in the y-dir,
    :return n_grid_z: number of grids in the z-dir,

    :return grid_vol: grid volume of shape (n_samples x n_grid_x x n_grid_y x n_grid_z)
    :return grid_xcoord: grid center x coordinate of shape (n_samples x n_grid_x x n_grid_y x n_grid_z)
    :return grid_ycoord: grid center y coordinate of shape (n_samples x n_grid_x x n_grid_y x n_grid_z)
    :return grid_zcoord: grid center z coordinate of shape (n_samples x n_grid_x x n_grid_y x n_grid_z)
    :return WI_inj: well index of shape (n_samples x n_grid_x x n_grid_y x n_grid_z)
    :return WI_prod: well index of shape (n_samples x n_grid_x x n_grid_y x n_grid_z)
    :return inactive_grid_filter: inactive grid filter (0: inactive; 1: active) of shape (n_samples x n_grid_x x n_grid_y x n_grid_z)
    :return dist_Prod_3d: grid distance to producer of shape (n_samples x n_grid_x x n_grid_y x n_grid_z),
    :return dist_Inj_3d: grid distance to injector of shape (n_samples x n_grid_x x n_grid_y x n_grid_z),

    :return pres_3d: pressure map  array of shape (n_samples x n_time_steps x n_grid_x x n_grid_y x n_grid_z),
    :return sG_3d: gas saturation map array of shape (n_samples x n_time_steps x n_grid_x x n_grid_y x n_grid_z),
    :return perm_3d: permeability map  array of shape (n_samples x n_time_steps x n_grid_x x n_grid_y x n_grid_z),
    :return poro_3d: porosity map  array of shape (n_samples x n_time_steps x n_grid_x x n_grid_y x n_grid_z),
    :return qInj_3d: surface injector rate map array of shape (n_samples x n_time_steps x n_grid_x x n_grid_y x n_grid_z),
    :return qProd_3d: surface producer rate map  array of shape (n_samples x n_time_steps x n_grid_x x n_grid_y x n_grid_z),
    :return BHP_Prod_3d: producer BHP map  array of shape (n_samples x n_time_steps x n_grid_x x n_grid_y x n_grid_z),

    :return timeStep_ts: time step array of shape (n_samples x n_time_steps),
    :return qInj_ts: injector rate array of shape (n_samples x n_time_steps),
    :return qProd_ts: producer rate array of shape (n_samples x n_time_steps),
    :return BHP_Prod_ts: producer BHP array of shape (n_samples x n_time_steps),
    :return BHP_Inj_ts: injector BHP array of shape (n_samples x n_time_steps),

    """

    final_pressures, final_gas_sat = [], []
    final_permeabilities, final_porosities = [], []
    final_timesteps, final_inj_rate_series, final_prod_rate_series, final_inj_bhp_series, final_prod_bhp_series = [], [], [], [], []
    final_grid_x, final_grid_y = [], []

    # get a list of all GEM file names
    files = glob.glob(f'{Path(data_path, "*.out")}')
    num_files = len(files)
    print(f'In total {len(files)} files.')

    # process each file in the list
    file_tag = {}
    for i in range(num_files):

        file = GEM_File(files[i], verb=0)

        msg = f'{Path(files[i]).name}, '

        # get a dictionary of I-direction permeabilities in 3D numpy array vs time
        permeabilities = file.get_variable(['I-direction','Permeabilities'], static=True)
        final_permeabilities.append(permeabilities[0.0])

        # msg += f'perm: {permeabilities[0.0][0][0][0]} md, '
        msg += f'perm: {np.mean(permeabilities[0.0]): .2e} md, '

        # get a dictionary of porosity in 3D numpy array vs time
        porosities = file.get_variable(['Current','Porosity'], static=True)
        final_porosities.append(porosities[0.0])

        # msg += f'poro: {porosities[0.0][0][0][0]}, '
        msg += f'poro: {np.mean(porosities[0.0]):.2e}, '

        # get a dictionary of pressures in 3D numpy array vs time
        pressures = file.get_variable(['Pressure','(psia)'])
        final_pressures.append([
            p for tstep, p in pressures.items() if tstep not in drop_timesteps
        ])

        # get a dictionary of 3D gas saturation field in 3D numpy array vs time
        saturations = file.get_variable(['Gas','Saturation'])
        final_gas_sat.append([
            sG for tstep, sG in saturations.items() if tstep not in drop_timesteps
        ])

        # get grid volume, grid_x, grid_y, grid_z
        if i == 0:

            # get a list of grid coordinate
            _, final_grid_x, final_grid_y, _ = file.get_grid_size()

            # get a list of all time steps
            timesteps = list(pressures.keys())
            final_timesteps = [t for t in timesteps if t not in drop_timesteps]

        # get a dictionary of CO2 injection rates vs time for well #1
        inj_rate_series = file.get_well_rates(1, ['Inst', 'Surface', 'Injection', 'Rates'],['Gas', 'MSCF/day'])
        final_inj_rate_series.append([
            qCO2_inj for tstep, qCO2_inj in inj_rate_series.items() if tstep not in drop_timesteps
        ])

        msg += f'qCO2_inj: {inj_rate_series[31.0]} MSCF/day, '

        # get a dictionary of water production rates vs time for well #2
        prod_rate_series = file.get_well_rates(2, ['Inst', 'Surface', 'Production','Rates'],['Water', 'STB/day'])
        final_prod_rate_series.append([
            bhp for tstep, bhp in prod_rate_series.items() if tstep not in drop_timesteps
        ])

        # get a dictionary of CO2 injection BHP vs time for well #1
        inj_bhp_series = file.get_well_rates(1, ['Well', 'Pressures'],['Bottom', 'Hole', 'psia'])
        final_inj_bhp_series.append([
            bhp for tstep, bhp in inj_bhp_series.items() if tstep not in drop_timesteps
        ])

        # get a dictionary of water production BHP vs time for well #2
        prod_bhp_series = file.get_well_rates(2, ['Well', 'Pressures'],['Bottom', 'Hole', 'psia'])
        final_prod_bhp_series.append([
            qW_prod for tstep, qW_prod in prod_bhp_series.items() if tstep not in drop_timesteps
        ])

        msg += f'bhp_prod: {prod_bhp_series[31.0]} psia'

        print(f'Processing file {i+1}/{num_files}: {msg}')

        file_tag.update({i: msg})

    # Convert all to numpy array
    # - Grid coordinates
    final_grid_x = np.array(final_grid_x)
    final_grid_y = np.array(final_grid_y)

    # - 3D geocellular properties
    final_gas_sat = np.array(final_gas_sat)
    final_pressures = np.array(final_pressures)
    final_permeabilities = np.array(final_permeabilities)
    final_porosities = np.array(final_porosities)

    # - time series data
    final_timesteps = np.array(final_timesteps)
    final_prod_bhp_series = np.array(final_prod_bhp_series)
    final_inj_bhp_series = np.array(final_inj_bhp_series)
    final_inj_rate_series = np.array(final_inj_rate_series)
    final_prod_rate_series = np.array(final_prod_rate_series)

    # - dimension parameters
    n_samples, n_time_steps, n_grid_x, n_grid_y, n_grid_z = final_gas_sat.shape

    data = {
        # - dimension parameters
        'n_samples': n_samples,
        'n_time_steps': n_time_steps,
        'n_grid_x': n_grid_x,
        'n_grid_y': n_grid_y,
        'n_grid_z': n_grid_z,
        # - 3D grid sizes and well index
        'grid_xcoord': final_grid_x,
        'grid_ycoord': final_grid_y,
        # - 3D geocellular properties
        'sG_3d': final_gas_sat,
        'pres_3d': final_pressures,
        'perm_3d': final_permeabilities,
        'poro_3d': final_porosities,
        # - time series data
        'timeStep_ts': final_timesteps,
        'qInj_ts': final_inj_rate_series,
        'qProd_ts': final_prod_rate_series,
        'BHP_Prod_ts': final_prod_bhp_series,
        'BHP_Inj_ts': final_inj_bhp_series,
    }

    return data, file_tag


if __name__ == '__main__':

    # EXAMPLE
    root_path = 'E:\\Research\\CCUS_DL_Gym\\cmg_output'
    h5_path = Path(root_path, 'cmg_gem_2D_cartesian_output.h5')
    file_tag_path = Path(root_path, 'file_tag.json')
    cpu = time.time()

    # Load the cmg gem output data
    data, file_tag = cmg_parser(root_path, drop_timesteps=[])

    # Save into h5 format
    save_to_h5(data, h5_path)
    with open(file_tag_path, 'w') as config_file:
        json.dump(file_tag, config_file, indent=4)

    # Load h5 file for machine learning training
    data = load_from_h5_general(h5_path)

    cpu = time.time() - cpu
    print(f'CPU cost = {cpu: .2e} sec.')


