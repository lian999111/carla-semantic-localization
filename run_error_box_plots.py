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
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
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
RECORDING_NAME = 'urban'
TEST_NAME = 'test_configs_of_factors'
NOISE_CONFIG_LABELS = ['w/o GNSS bias', 'w/ GNSS bias']
SW_CONFIG_LABELS = ['GNSS+Lane', 'GNSS+Lane+Pole', 'GNSS+Lane+Pole+Stop']

FIG_NAME = 'urban'

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

abs_lon_err_dict = {}
abs_lat_err_dict = {}
abs_yaw_err_dict = {}

for noise_config_file_name in noise_config_file_names:
    noise_config = os.path.splitext(noise_config_file_name)[0]
    noise_config = remove_prefix(noise_config, 'n_')

    print('')

    for sw_config_file_name in sw_config_file_names:
        sw_config = os.path.splitext(sw_config_file_name)[0]
        sw_config = remove_prefix(sw_config, 'sw_')

        results = results_in_all_tests[noise_config][sw_config]
        lon_errs = results['lon_errs']
        lat_errs = results['lat_errs']
        yaw_errs = results['yaw_errs']

        if sw_config not in abs_lon_err_dict:
            abs_lon_err_dict[sw_config] = {}
            abs_lat_err_dict[sw_config] = {}
            abs_yaw_err_dict[sw_config] = {}

        lon_abs_errs = np.abs(np.asarray(lon_errs))
        lat_abs_errs = np.abs(np.asarray(lat_errs))
        yaw_abs_errs = np.abs(np.asarray(yaw_errs))

        abs_lon_err_dict[sw_config][noise_config] = lon_abs_errs
        abs_lat_err_dict[sw_config][noise_config] = lat_abs_errs
        abs_yaw_err_dict[sw_config][noise_config] = yaw_abs_errs

        print('{}, {}:'.format(noise_config, sw_config))
        print('Number of data points: {}'.format(len(lon_abs_errs)))
        print(' CPU time mean: {:.6f}'.format(np.mean(results['cpu_times'])))
        print(' CPU time median: {:.6f}'.format(np.median(results['cpu_times'])))
        print('  Lon mean abs error: {:.2f}'.format(lon_abs_errs.mean()))
        print('  Lon median abs error: {:.2f}'.format(np.median(lon_abs_errs)))
        # print('  Lon RMSE: {}'.format(np.sqrt(np.mean(lon_abs_errs**2))))
        print('  Lat mean abs error: {:.2f}'.format(lat_abs_errs.mean()))
        print('  Lat median abs error: {:.2f}'.format(np.median(lat_abs_errs)))
        # print('  Lat RMSE: {}'.format(np.sqrt(np.mean(lat_abs_errs**2))))
        print('  Yaw mean abs error: {:.3f}'.format(yaw_abs_errs.mean()))
        print('  Yaw median abs error: {:.3f}'.format(np.median(yaw_abs_errs)))
        # print('  Yaw RMSE: {}'.format(np.sqrt(np.mean(yaw_abs_errs**2))))

# Number of configs
# Used for boxplot spacing
num_sw_configs = len(sw_config_file_names)
num_noise_configs = len(noise_config_file_names)

if len(NOISE_CONFIG_LABELS) != num_noise_configs:
    raise RuntimeError('Number of noise config labels should be: {} \
                       \nNoise config names are: {}'.format(num_noise_configs, noise_config_file_names))


flier = dict(markeredgecolor='gray', marker='+')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axs = plt.subplots(1, 3)
for idx, (lon_errs, lat_errs, yaw_errs) in enumerate(zip(abs_lon_err_dict.values(),
                                                         abs_lat_err_dict.values(),
                                                         abs_yaw_err_dict.values())):
    list_of_lon_errs = [err for err in lon_errs.values()]
    list_of_lat_errs = [err for err in lat_errs.values()]
    list_of_yaw_errs = [err for err in yaw_errs.values()]

    positions = np.linspace(0, (num_sw_configs+1) *
                            (num_noise_configs-1), num_noise_configs, dtype=int)
    positions += idx

    # Box plot of a sw config under different noise configs
    bp = axs[0].boxplot(
        list_of_lon_errs, positions=positions, flierprops=flier)
    set_box_color(bp, colors[idx])
    axs[0].plot([], c=colors[idx], label=SW_CONFIG_LABELS[idx])

    # Box plot of a sw config under different noise configs
    bp = axs[1].boxplot(
        list_of_lat_errs, positions=positions, flierprops=flier)
    set_box_color(bp, colors[idx])
    axs[1].plot([], c=colors[idx], label=SW_CONFIG_LABELS[idx])

    # Box plot of a sw config under different noise configs
    bp = axs[2].boxplot(
        list_of_yaw_errs, positions=positions, flierprops=flier)
    set_box_color(bp, colors[idx])
    axs[2].plot([], c=colors[idx], label=SW_CONFIG_LABELS[idx])


axs[0].set_ylabel('abs. longitudinal error [m]')
axs[1].set_ylabel('abs. lateral error [m]')
axs[2].set_ylabel('abs. yaw error [rad]')

axs[0].set_ylim((-0.1, 4))
axs[1].set_ylim((-0.1, 4))
axs[2].set_ylim((-0.05, 0.5))

first_mid = (num_sw_configs-1)/2
tick_positions = first_mid + np.arange(num_noise_configs)*(num_sw_configs+1)
for ax in axs:
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(NOISE_CONFIG_LABELS)
    ax.yaxis.grid()

fig.set_size_inches(10, 4)
axs[0].legend(framealpha=1.0, fontsize=10,
              bbox_to_anchor=(0, 1), loc='lower left')
fig.tight_layout()
plt.show()

if FIG_NAME:
    fig.savefig(FIG_NAME+'_box_plot.svg', bbox_inches='tight')


# %%
