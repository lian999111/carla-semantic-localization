"""This script runs predefined localization tests automatically.

Ref: 
https://medium.com/@leportella/how-to-run-parallel-processes-8939dafae81e
https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
"""

import os
import subprocess
from multiprocessing import Pool

import yaml


with open('settings/tests/scenarios.yaml', 'r') as f:
    scenarios = yaml.safe_load(f)

BASE_RECORDING_PATH = 'recordings/'
BASE_CONFIG_PATH = 'settings/tests'

BASE_COMMAND = 'python -O sliding_window_localization.py'

# Create a process pool of 2
# On my laptop, using more processes tends to make CARLA crash
# But you can try if you have better luck
pool = Pool(processes=2)
args = []

# Loop over test scenarios
for test_name, test_configs in scenarios.items():

    # Loop over recording names (highway & urban)
    for recording_name, params in test_configs.items():
        recording_dir = os.path.join(BASE_RECORDING_PATH,
                                     recording_name)

        # Get noise and sliding window configs
        noise_configs = params['noise_configs']
        sw_configs = params['sw_configs']

        # Loop over noise configs
        child_procs = []
        for noise_config in noise_configs:
            noise_config_path = os.path.join(BASE_CONFIG_PATH,
                                             test_name,
                                             recording_name,
                                             noise_config)

            # Loop over sliding window configs
            for sw_config in sw_configs:
                print('Running test: {}'.format(test_name))
                print('  Using {} recording'.format(recording_name))
                print('    Noise config: {}'.format(noise_config))
                print('       SW config: {}'.format(sw_config))

                sw_config_path = os.path.join(BASE_CONFIG_PATH,
                                              test_name,
                                              recording_name,
                                              sw_config)

                save_path = os.path.join(test_name,
                                         os.path.splitext(noise_config)[0],
                                         os.path.splitext(sw_config)[0])

                args_to_localization = ['python', '-O', 'sliding_window_localization.py',
                                        recording_dir, sw_config_path,
                                        '-n', noise_config_path,
                                        '-s', save_path]

                args.append(args_to_localization)

pool.map(subprocess.call, args)
pool.close()
pool.join()
