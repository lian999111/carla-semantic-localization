"""This script runs specified evaluations automatically."""
# %% Imports
import os
import argparse
import pickle

import yaml
import numpy as np
import matplotlib.pyplot as plt

from carlasim.utils import TrafficSignType
import localization.eval.utils as evtools


# %% ############### Set matplotlib's format ###############
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
params = {'text.latex.preamble': [
    r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

# %%


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def get_subdir_names(directory):
    """
    # Ref: https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory
    """
    return next(os.walk(directory))[1]


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


# %%  ############### Set directories manually ###############
RECORDING_NAME = 'highway'
TEST_NAME = 'test_configs_of_factors'
SW_CONFIG_LABELS = ['GNSS-only', 'GNSS+Lane']

FIG_NAME = 'hw'
SAVE_CPU_FIG = False

BINS = 40

recording_dir = os.path.join('recordings', RECORDING_NAME)
test_dir = os.path.join(recording_dir, 'results', TEST_NAME)
print('Recording: {}'.format(RECORDING_NAME))
print('Test Name: {}'.format(TEST_NAME))

# %% ############### Load carla simulation configs ###############
path_to_config = os.path.join(recording_dir,
                              'settings/config.yaml')
with open(path_to_config, 'r') as f:
    carla_config = yaml.safe_load(f)

# %% ############### Load results with same noise level ###############
# Get all noise level subdirectories under the test folder
noise_configs = get_subdir_names(test_dir)

results_in_all_tests = {}
# Loop over noise levels
for noise_config in noise_configs:
    # Create a dict for this noise level
    shorten_noise_level_name = remove_prefix(noise_config, 'n_')
    results_in_all_tests[shorten_noise_level_name] = {}

    # Find sw configs in this noise level
    noise_level_dir = os.path.join(test_dir, noise_config)
    sw_configs = get_subdir_names(noise_level_dir)
    # Loop over sw configs
    for sw_config in sw_configs:
        result_dir = os.path.join(noise_level_dir, sw_config)
        path_to_result_file = os.path.join(result_dir,
                                           'results.pkl')
        with open(path_to_result_file, 'rb') as f:
            localization_results = pickle.load(f)

        shorten_config_name = remove_prefix(sw_config, 'sw_')
        results_in_all_tests[shorten_noise_level_name][shorten_config_name] = localization_results

# %% ############### Evaluate errors across all configs ###############

# Use the order of configs defined in scenarios.yaml
with open('settings/tests/scenarios.yaml', 'r') as f:
    scenarios = yaml.safe_load(f)

noise_config_file_names = scenarios[TEST_NAME][RECORDING_NAME]['noise_configs']
sw_config_file_names = scenarios[TEST_NAME][RECORDING_NAME]['sw_configs']

lon_err_dict = {}
lat_err_dict = {}
yaw_err_dict = {}
cpu_time_dict = {}

for noise_config_file_name in noise_config_file_names:
    noise_config = os.path.splitext(noise_config_file_name)[0]
    noise_config = remove_prefix(noise_config, 'n_')

    lon_err_dict[noise_config] = {}
    lat_err_dict[noise_config] = {}
    yaw_err_dict[noise_config] = {}
    cpu_time_dict[noise_config] = {}

    for sw_config_file_name in sw_config_file_names:
        sw_config = os.path.splitext(sw_config_file_name)[0]
        sw_config = remove_prefix(sw_config, 'sw_')

        results = results_in_all_tests[noise_config][sw_config]
        lon_errs = results['lon_errs']
        lat_errs = results['lat_errs']
        yaw_errs = results['yaw_errs']
        cpu_time = results['cpu_times']

        lon_err_dict[noise_config][sw_config] = np.asarray(lon_errs)
        lat_err_dict[noise_config][sw_config] = np.asarray(lat_errs)
        yaw_err_dict[noise_config][sw_config] = np.asarray(yaw_errs)
        cpu_time_dict[noise_config][sw_config] = np.asarray(cpu_time)*1000

        print('{}, {}:'.format(noise_config, sw_config))
        print('Number of data points: {}'.format(len(lon_errs)))
        print(' CPU time mean: {:.6f}'.format(np.mean(cpu_time)))
        print(' CPU time median: {:.6f}'.format(
            np.median(results['cpu_times'])))
        print('  Lon mean error: {:.2f}'.format(np.mean(lon_errs)))
        print('  Lon median error: {:.2f}'.format(np.median(lon_errs)))
        # print('  Lon RMSE: {}'.format(np.sqrt(np.mean(lon_errs**2))))
        print('  Lat mean error: {:.2f}'.format(np.mean(lat_errs)))
        print('  Lat median error: {:.2f}'.format(np.median(lat_errs)))
        # print('  Lat RMSE: {}'.format(np.sqrt(np.mean(lat_errs**2))))
        print('  Yaw mean error: {:.3f}'.format(np.mean(yaw_errs)))
        print('  Yaw median error: {:.3f}'.format(np.median(yaw_errs)))
        # print('  Yaw RMSE: {}'.format(np.sqrt(np.mean(yaw_abs_errs**2))))

# Number of configs
num_sw_configs = len(sw_config_file_names)
num_noise_configs = len(noise_config_file_names)

# if len(NOISE_CONFIG_LABELS) != num_noise_configs:
#     raise RuntimeError('Number of noise config labels should be: {} \
#                        \nNoise config names are: {}'.format(num_noise_configs, noise_config_file_names))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

figs = []
figs_cpu = []
# Loop over different noise levels
for lon_err_arrs, lat_err_arrs, yaw_err_arrs, cpu_time_arrs in zip(lon_err_dict.values(),
                                                                   lat_err_dict.values(),
                                                                   yaw_err_dict.values(),
                                                                   cpu_time_dict.values()):
    fig, axs = plt.subplots(3, 1)
    fig_cpu, ax_cpu = plt.subplots(1, 1, figsize=(7,2))

    first_lon_err_arr = list(lon_err_arrs.values())[0]
    weights = []
    for lon_err_arr in lon_err_arrs.values():
        weights.append(np.ones_like(lon_err_arr) / len(lon_err_arr))
    # weights = np.ones_like(lon_errs) / len(lon_errs)
    axs[0].hist(lon_err_arrs.values(), bins=BINS, range=(-4, 4),
                weights=weights, label=SW_CONFIG_LABELS)
    axs[0].set_ylabel('probability')
    axs[0].set_xlabel('longitudinal error [m]')

    # weights = np.ones_like(lat_errs) / len(lat_errs)
    axs[1].hist(lat_err_arrs.values(), bins=BINS, range=(-4, 4),
                weights=weights, label=SW_CONFIG_LABELS)
    axs[1].set_ylabel('probability')
    axs[1].set_xlabel('lateral error [m]')

    # weights = np.ones_like(yaw_errs) / len(yaw_errs)
    axs[2].hist(yaw_err_arrs.values(), bins=BINS, range=(-0.5, 0.5),
                weights=weights, label=SW_CONFIG_LABELS)
    axs[2].set_ylabel('probability')
    axs[2].set_xlabel('yaw error [rad]')

    ax_cpu.hist(cpu_time_arrs.values(), bins=BINS, range=(0, 1200),
                weights=weights, label=SW_CONFIG_LABELS)
    ax_cpu.set_ylabel('probability')
    ax_cpu.set_xlabel('cpu time [ms]')
    ax_cpu.yaxis.grid()
    ax_cpu.legend()
    
    for ax in axs:
        ax.yaxis.grid()

    fig_cpu.tight_layout()
    fig.tight_layout()
    axs[0].legend(framealpha=1.0, fontsize=10,
              bbox_to_anchor=(0, 1), loc='lower left')

    figs.append(fig)
    figs_cpu.append(fig_cpu)


if FIG_NAME:
    for fig, noise_name in zip(figs, lon_err_dict.keys()):
        fig.savefig(FIG_NAME+'_'+noise_name+'_histo.svg', bbox_inches='tight')
if SAVE_CPU_FIG and FIG_NAME:
    for fig_cpu, noise_name in zip(figs_cpu, lon_err_dict):
        fig_cpu.savefig(FIG_NAME+'_'+noise_name+'_cpu_time_histo.svg', bbox_inches='tight')

plt.show()
# %%
