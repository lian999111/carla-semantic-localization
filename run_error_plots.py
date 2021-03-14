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


# %%  ############### Set parameters ###############
RECORDING_NAME = 'urban'
TEST_NAME = 'test_null_hypo'
NOISE_LEVEL = 'n_high_fc'
SW_CONFIG = 'sw_no_null'
FIG_SIZE = 5

FIG_NAME = 'urban_no_null'

# %% ############### Set matplotlib's format ###############
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
params = {'text.latex.preamble' : r'\usepackage{siunitx} \usepackage{amsmath}'}
plt.rcParams.update(params)

# %% ############### Create directories ###############
recording_dir = os.path.join('recordings', RECORDING_NAME)
test_dir = os.path.join(recording_dir, 'results', TEST_NAME)
noise_level_dir = os.path.join(test_dir, NOISE_LEVEL)
result_dir = os.path.join(noise_level_dir, SW_CONFIG)

# %% ############### Load carla simulation configs ###############
path_to_config = os.path.join(recording_dir,
                              'settings/config.yaml')
with open(path_to_config, 'r') as f:
    carla_config = yaml.safe_load(f)

# %% ############### Load results ###############
path_to_result_file = os.path.join(result_dir,
                                   'results.pkl')

with open(path_to_result_file, 'rb') as f:
    localization_results = pickle.load(f)

loc_gt_seq = localization_results['loc_gt_seq']
ori_gt_seq = localization_results['ori_gt_seq']
pose_estimations = localization_results['pose_estimations']
cpu_times = localization_results['cpu_times']
lon_errs = localization_results['lon_errs']
lat_errs = localization_results['lat_errs']
yaw_errs = localization_results['yaw_errs']

# Absolute errors
abs_lon_errs = np.abs(np.asarray(lon_errs))
abs_lat_errs = np.abs(np.asarray(lat_errs))
abs_yaw_errs = np.abs(np.asarray(yaw_errs))

# %% ############### Load pole map for visualization ###############
with open(os.path.join(recording_dir, 'pole_map.pkl'), 'rb') as f:
    pole_map = pickle.load(f)

# Retrieve pole ground truth
sign_pole_coords = np.array([[pole.x, pole.y] for pole in pole_map if (
    pole.type != TrafficSignType.Unknown and pole.type != TrafficSignType.RSStop)])
general_pole_coords = np.array([[pole.x, pole.y] for pole in pole_map if
                                pole.type == TrafficSignType.Unknown])

# %% ############### Load map image ###############
dirname = os.path.join("cache", "map_images")
filename = carla_config['world']['map'] + '.jpg'
full_path = os.path.join(dirname, filename)
# Load cached map image
map_image = plt.imread(full_path)

# Load map info of how to show a pose on the map image
info_filename = carla_config['world']['map'] + '_info.yaml'
info_full_path = str(os.path.join(dirname, info_filename))
with open(info_full_path, 'r') as f:
    map_info = yaml.safe_load(f)


# %% ############### Get test path info ###############
print('Length of data: {}'.format(len(loc_gt_seq)))
distance = 0.0
for p1, p2 in zip(loc_gt_seq[:-1], loc_gt_seq[1:]):
    distance += np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
print('Distance of path: {}'.format(distance))


# %% ############### Result visualization ###############
# Prepare local map as background
local_map_image, extent = evtools.get_local_map_image(
    loc_gt_seq, map_image, map_info, pose_estimations=pose_estimations)

if RECORDING_NAME == 'highway':
    ## Longitudinal error ##
    lon_err_fig, lon_err_ax = evtools.gen_colored_error_plot('abs. longitudinal error [m]',
                                                             abs_lon_errs, 3.0,
                                                             loc_gt_seq, pose_estimations,
                                                             sign_pole_coords, general_pole_coords,
                                                             local_map_image, extent, FIG_SIZE, zoom_in=True)

    ## Lateral error ##
    lat_err_fig, lat_err_ax = evtools.gen_colored_error_plot('abs. lateral error [m]',
                                                             abs_lat_errs, 1.0,
                                                             loc_gt_seq, pose_estimations,
                                                             sign_pole_coords, general_pole_coords,
                                                             local_map_image, extent, FIG_SIZE, zoom_in=True)

    ## Yaw error ##
    yaw_err_fig, yaw_err_ax = evtools.gen_colored_error_plot('abs. yaw error [m]',
                                                             abs_yaw_errs, 0.5,
                                                             loc_gt_seq, pose_estimations,
                                                             sign_pole_coords, general_pole_coords,
                                                             local_map_image, extent, FIG_SIZE, zoom_in=True)
else:
    ## Longitudinal error ##
    lon_err_fig, lon_err_ax = evtools.gen_colored_error_plot('abs. longitudinal error [m]',
                                                             abs_lon_errs, 3.0,
                                                             loc_gt_seq, pose_estimations,
                                                             sign_pole_coords, general_pole_coords,
                                                             local_map_image, extent, FIG_SIZE)

    ## Lateral error ##
    lat_err_fig, lat_err_ax = evtools.gen_colored_error_plot('abs. lateral error [m]',
                                                             abs_lat_errs, 1.0,
                                                             loc_gt_seq, pose_estimations,
                                                             sign_pole_coords, general_pole_coords,
                                                             local_map_image, extent, FIG_SIZE)

    ## Yaw error ##
    yaw_err_fig, yaw_err_ax = evtools.gen_colored_error_plot('abs. yaw error [rad]',
                                                             abs_yaw_errs, 0.5,
                                                             loc_gt_seq, pose_estimations,
                                                             sign_pole_coords, general_pole_coords,
                                                             local_map_image, extent, FIG_SIZE)

if FIG_NAME:
    lon_err_fig.savefig(FIG_NAME+'_lon_err.svg', dpi=500, bbox_inches='tight')
    lat_err_fig.savefig(FIG_NAME+'_lat_err.svg', dpi=500, bbox_inches='tight')
    yaw_err_fig.savefig(FIG_NAME+'_yaw_err.svg', dpi=500, bbox_inches='tight')

cpu_time_fig, cpu_time_ax = plt.subplots(1, 1)
cpu_time_ax.boxplot(cpu_times)
cpu_time_ax.set_ylabel('cpu time [sec]')

# plt.show()

# %%
