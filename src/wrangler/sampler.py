###############################################################################################
# Generate CMG input files and make it ready for simulation
# Developer: Bicheng Yan
# 4/13/2022
# DSFT, KAUST
###############################################################################################

import json
import time
import numpy as np
from pathlib import Path

from wrangler.cmg_gem import cmg_parser
from wrangler.cmg_parallel import parallel_run_sim, cmg_engine
from wrangler.data_parser import save_to_h5, load_from_h5_general

def get_phi(perm):
    """An assumed correlation between permeability and porosity."""
    return 0.05 * np.log10(perm) + 0.1


def generate_sample_values():

    # Generate value samples
    permeabilities = np.array([10, 25, 50])
    porosities = get_phi(permeabilities)
    injection_rates = 1e6 * np.arange(1, 4, 1)
    nsamples = len(permeabilities) * len(injection_rates)
    # Mutate the samples
    samples = np.zeros(shape=(nsamples, 3), dtype=float)
    isample = 0
    for iperm in permeabilities:
        for irate in injection_rates:
            iporo = get_phi(iperm)
            samples[isample, 0] = iperm
            samples[isample, 1] = iporo
            samples[isample, 2] = irate
            isample += 1

    return samples


def generate_sim_input_files(template_file, samples):

    input_file_paths = []

    # Load template file
    template = template_file.read_text()
    file_prefix = template_file.name.split('.')[0]
    file_postfix = template_file.name.split('.')[1]
    for isample in range(samples.shape[0]):
        ifile = Path(template_file.parent, f'{file_prefix}_{isample}.{file_postfix}')
        iperm = samples[isample, 0]
        iporo = samples[isample, 1]
        irate = samples[isample, 2]
        ifile_text = template
        ifile_text = ifile_text.replace("_%%permeability%%_", f"{iperm: .1f}")
        ifile_text = ifile_text.replace("_%%porosity%%_", f"{iporo: .3f}")
        ifile_text = ifile_text.replace("_%%injection_rate%%_", f"{irate: .1e}")

        ifile.write_text(ifile_text)

        # Collect input path
        input_file_paths.append(ifile)

    return input_file_paths


if __name__ == '__main__':

    # CMG input file template
    template_file = Path(f'E:\\Research\\CCUS_DL_Gym\\cmg_output\\Toy2D_case.DAT')

    # CMG gem executable file
    exe_path = Path('C:\\Program Files\\CMG\\GEM\\2020.11\\Win_x64\\EXE\\gm202011.exe')

    # Get samples
    cpu = time.time()
    samples = generate_sample_values()

    # Generate cmg input files based on template and sample values
    input_file_paths = generate_sim_input_files(template_file, samples)
    cpu = time.time() - cpu
    print(f'Sampling CPU cost = {cpu: .2e} sec.')

    # Run the simulation
    cpu = time.time()
    parallel_run_sim(exe_path, input_file_paths, simulator=cmg_engine)
    cpu = time.time() - cpu
    print(f'Simulation CPU cost = {cpu: .2e} sec.')

    # EXAMPLE
    root_path = template_file.parent
    h5_path = Path(root_path, 'cmg_gem_2D_cartesian_output.h5')
    file_tag_path = Path(root_path, 'file_tag.json')
    cpu = time.time()

    # Load the cmg gem output data
    data, file_tag = cmg_parser(root_path, drop_timesteps=[])

    # Save into h5 format
    save_to_h5(data, h5_path)
    with open(file_tag_path, 'w') as config_file:
        json.dump(file_tag, config_file, indent=4)
    cpu = time.time() - cpu
    print(f'Data Collection CPU cost = {cpu: .2e} sec.')

    # Load h5 file for machine learning training
    data = load_from_h5_general(h5_path)

