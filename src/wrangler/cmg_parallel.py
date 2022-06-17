
import subprocess
from pathlib import Path
from joblib import Parallel, delayed

def cmg_engine(exe_path, input_path):
    """Run CMG Gem,
    parameters:
        exe_path = simulator executable path in pathlib.Path format;
        input_path = dek file path in pathlib.Path format
    ."""

    # Define output path based on CMG naming rules
    output_path = Path(input_path.parent, input_path.name.split('.')[0])

    # Go with subprocess
    cmd = f'"{exe_path}" -f "{input_path}" -o "{output_path}"'
    try:
        # If simulation time go beyond 600 seconds, kill it
        proc = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, timeout=3600)
    except subprocess.TimeoutExpired:
        print('Timeout: CMG job killed!')


def parallel_run_sim(exe_path, dek_paths, simulator):
    """Parallely run a simulator for reservoir simulation."""

    # Parallel version: func = cmg_gem, defined above.
    Parallel(n_jobs=len(dek_paths))(delayed(simulator)(exe_path, idek) for idek in dek_paths)

    return


if __name__ == '__main__':

    exe_path = Path('C:\\Program Files\\CMG\\GEM\\2020.11\\Win_x64\\EXE\\gm202011.exe')
    input_path = Path('C:\\Users\\YANB0A\\Desktop\\cmg_test\\gmghg001.dat')

    # Test single run
    # cmg_gem(exe_path, input_path)

    # Test parallel run
    dek_paths = [
        Path('C:\\Users\\YANB0A\\Desktop\\cmg_test\\gmghg001.dat'),
        Path('C:\\Users\\YANB0A\\Desktop\\cmg_test\\gmghg002.dat')
    ]
    parallel_run_sim(exe_path, dek_paths, simulator=cmg_engine)

