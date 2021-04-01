This folder contains helper scripts and Jupyter notebooks. They can be divided into 2 categories: CARLA-related and Thesis-related. Their usages are briefly described here.

__CARLA-related__:
- __explor_map.py__: Use this script to manually wander around a map with a randomly spawned car. It is modified from a CARLA's example which is under the MIT license. The modification extends the visualization to also show the 4 lane boundaries in the neightborhood. This is intend to be a tool to find inconsistencies between the lane boundaries in the underlying map and the rendering.

- __show_all_landmarks.py__: As its names suggests, it visualizes all landmark objects in CARLA's rendering. A landmark object corresponds to a traffic sign defined in the underlying map.

- __show_waypoints.py__: Shows all waypoints in CARLA's rendering.

__Thesis-related__:

These scripts or notebooks are to generate images for thesis' use. When a manual setting of __TEST_NAME__, __NOISE_LEVEL__, or __SW_CONFIG__ (SW stands for slidig window) is required, refer to [scenarios.yaml](../settings/tests/scenarios.yaml) for the configuration names. Such settings can be found under the comment title ############### Set directories manually ###############. The order of image labels (if required) should follow the order of configurations defined in scenarios.yaml.
- __gen_error_box_plots.py__: Generates error box plots of a test.

- __gen_error_box_plots_null_test.py__: Generates error box plots of null hypothesis test in the thesis. It is separated from the above one because the labelling logic is different.

- __gen_error_histo_plots.py__: Generates error histograms of a test.

- __gen_error_plots.py__: Generates color-coded error plots of a test.

- __gen_gt_plots.py__: Generates ground truth path plots of a scenario recording.

- __gen_plot_for_sem_test.py__: Generates a comparison plot of a semantic information test.

The following two Jupyter notebooks relies on the __urban__ scenario to be recorded and saved first. They are ugly modified from the original detection simulation modules, but they get the job done.
- __visualize_lane_detection.ipynb__: Generate visualizations of lane detection process.

- __visualize_pole_detection.ipynb__: Generate visualizations of pole detection process.

__Misc__:
- __show_range_of_pxs.ipynb__: This notebook is used to investigate the relationship between the pixels along the vertical center line in the image and their corresponding depth (x-coordinate in the front bumper frame).
