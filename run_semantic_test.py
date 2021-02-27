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
RECORDING_NAME = 's3'
TEST_NAME = 'test_semantic'
NOISE_LEVEL = 'n_neg_x_gnss_bias'
SW_SEMANTIC_ON = 'sw_semantic_on'
SW_SEMANTIC_OFF = 'sw_semantic_off'

FIG_SIZE = 5
LEGEND_FONT_SIZE = FIG_SIZE+5

FIG_NAME = 'urban'

# %% ############### Set matplotlib's format ###############
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=FIG_SIZE+7)
params = {'text.latex.preamble' : [r'\usepackage{siunitx}', r'\usepackage{amsmath}']}
plt.rcParams.update(params)

# %% ############### Create directories ###############
recording_dir = os.path.join('recordings', RECORDING_NAME)
test_dir = os.path.join(recording_dir, 'results', TEST_NAME)
noise_level_dir = os.path.join(test_dir, NOISE_LEVEL)
result_sem_on_dir = os.path.join(noise_level_dir, SW_SEMANTIC_ON)
result_sem_off_dir = os.path.join(noise_level_dir, SW_SEMANTIC_OFF)

# %% ############### Load carla simulation configs ###############
path_to_config = os.path.join(recording_dir,
                              'settings/config.yaml')
with open(path_to_config, 'r') as f:
    carla_config = yaml.safe_load(f)

# %% ############### Load results ###############
# For semantic on
path_to_sem_on = os.path.join(result_sem_on_dir,
                                   'results.pkl')

with open(path_to_sem_on, 'rb') as f:
    localization_results_sem_on = pickle.load(f)

loc_gt_seq = localization_results_sem_on['loc_gt_seq']
ori_gt_seq = localization_results_sem_on['ori_gt_seq']
pose_estimations_sem_on = localization_results_sem_on['pose_estimations']

# For semantic off
path_to_sem_off = os.path.join(result_sem_off_dir,
                                   'results.pkl')

with open(path_to_sem_off, 'rb') as f:
    localization_results_sem_off = pickle.load(f)

pose_estimations_sem_off = localization_results_sem_off['pose_estimations']

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
    loc_gt_seq, map_image, map_info, pose_estimations=pose_estimations_sem_off)

# Generate ground truth plot
gt_fig, gt_ax = evtools.gen_gt_path_plot(loc_gt_seq, sign_pole_coords, general_pole_coords,
                                         local_map_image, extent, FIG_SIZE)

x_estimations = [pose[0] for pose in pose_estimations_sem_off]
y_estimations = [pose[1] for pose in pose_estimations_sem_off]
gt_ax.plot(x_estimations, y_estimations, '-', label='sem. off')

x_estimations = [pose[0] for pose in pose_estimations_sem_on]
y_estimations = [pose[1] for pose in pose_estimations_sem_on]
gt_ax.plot(x_estimations, y_estimations, '-', label='sem. on')

gt_ax.set_xlim([extent[0], extent[1]])
gt_ax.set_ylim([extent[2], extent[3]])


legend = gt_ax.legend(framealpha=1.0, edgecolor='none', fontsize=LEGEND_FONT_SIZE)    

if FIG_NAME:
    gt_fig.savefig(FIG_NAME+'.svg', dpi=600, bbox_inches='tight')

plt.show()

# %%
