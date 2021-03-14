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
RECORDING_NAME = 'highway'
FIG_SIZE = 5
LEGEND_FONT_SIZE = 14

FIG_NAME = RECORDING_NAME

# %% ############### Set matplotlib's format ###############
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
params = {'text.latex.preamble' : r'\usepackage{siunitx} \usepackage{amsmath}'}
plt.rcParams.update(params)

# %% ############### Create directories ###############
recording_dir = os.path.join('recordings', RECORDING_NAME)

# %% ############### Load carla simulation configs ###############
path_to_config = os.path.join(recording_dir,
                              'settings/config.yaml')
with open(path_to_config, 'r') as f:
    carla_config = yaml.safe_load(f)

# %% ############### Load gt data ###############
path_to_gt_data = os.path.join(recording_dir,
                               'gt_data.pkl')
with open(path_to_gt_data, 'rb') as f:
    gt_data = pickle.load(f)

raxle_locations = gt_data['seq']['pose']['raxle_location']

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
print('Length of data: {}'.format(len(raxle_locations)))
distance = 0.0
for p1, p2 in zip(raxle_locations[:-1], raxle_locations[1:]):
    distance += np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
print('Distance of path: {}'.format(distance))


# %% ############### Result visualization ###############
# Prepare local map as background
local_map_image, extent = evtools.get_local_map_image(
    raxle_locations, map_image, map_info)

if RECORDING_NAME == 'highway':
    gt_plot_fig, gt_plot_ax = evtools.gen_gt_path_plot(raxle_locations,
                                                       sign_pole_coords, general_pole_coords,
                                                       local_map_image, extent, FIG_SIZE, 
                                                       add_legend=True, zoom_in=True)
else:
    gt_plot_fig, gt_plot_ax = evtools.gen_gt_path_plot(raxle_locations,
                                                       sign_pole_coords, general_pole_coords,
                                                       local_map_image, extent, FIG_SIZE,
                                                       add_legend=True)
# Overwrite the legend format
gt_plot_ax.legend(framealpha=1.0, edgecolor='none', fontsize=LEGEND_FONT_SIZE)

if FIG_NAME:
    gt_plot_fig.savefig(FIG_NAME+'_gt_path.svg', dpi=500)
plt.show()

# %%
