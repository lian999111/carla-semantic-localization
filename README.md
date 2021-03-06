# SMMPDA

This repo implements Semantic Max-mixture Probabilistic Data Association (SMMPDA) localization using the miniSAM library. Simulations are conducted with CARLA. Ubuntu 18.04 LTS has been used to develop this project. The publication to ITSC 2021 based on this work is in progress together with the RST Lab at TU Dortmund.

SMMPDA is a localization algorithm based on sliding window factor graphs combined with a PDA-like data association scheme. The following animation gives a sense of the localization result in CARLA simulations. The meanings of the elements are:
- Triangles with ellipses: The estimated poses in the sliding window. These two examples use the window size of 10.
- Blue curve: The ground truth trajectory
- Cross: The ground truth position at the current time step.
- Blue and red dots: Pole objects in the map. (blue: general pole. red: a pole of a traffic sign)
- Jumping dot: The GNSS reading. Biases are added to degrade its reliablity in both cases.
- Orange & green curves: The detected lane boundaries.
- Lines connecting to hollow dots: The detected pole objects. The hollow dot ahead of the pose triangle is the camera. (blue: detected as general pole. red: detected as a pole of a traffic sign)

Note: The implementation also uses stop line detections (enmulated by road surface stop signs in CARLA), which are not visualized.

In the first highway case, with semantic lane boundary measurements, SMMPDA is able to recover from a wrong initial belief in the wrong lane resulting from laterally-biased GNSS measurements and achieve a lane-level accuracy. SMMPDA is capable of reinitialization when it detects previous beliefs were potentially wrong, which takes effect after the lane change in this case. (Poles not used in this case)
<p align="center">
  <img src="for_readme/localization_hw.gif" />
</p>

In the second urban case, SMMPDA results in fairly accurate localization even with biases in the longitudinal and lateral directions.
<p align="center">
  <img src="for_readme/localization_urban.gif" />
</p>

## Environment Setup
Using miniconda is recommended beause this repo is developed this way. The repo comes with an [environment.yml](environment.yml) file that facilitates setting up the environment.

To create the same conda environment with the default name "__smmpda__":
```
conda env create -f environment.yml
```
Refer to [Creating an environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more info.

## Dependencies
This project depends on CARLA and miniSAM, which need to be installed manually in addition to the conda environment. Follow their official guides for proper installation.

[CARLA](http://carla.org/)  
The version __0.9.10__ is used throughout this repo. The corresponding .egg file has be included in the root directory of this repo. To upgrade, replace the .egg file with a newer version. CARLA suggests two ways to import PythonAPIs as written in [here](https://carla.readthedocs.io/en/latest/build_system/). The first inelegant method by adding the .egg file to the system path is used because _easy_install_ seems to be deprecated ([ref1](https://setuptools.readthedocs.io/en/latest/deprecated/easy_install.html), [ref2](https://workaround.org/easy-install-debian)).

[miniSAM](https://minisam.readthedocs.io/index.html)  
Follow the instructions in [C++ Compile and Installation](https://minisam.readthedocs.io/install.html) and [Python Installation](https://minisam.readthedocs.io/install_python.html) to install the miniSAM library. Some additional dependencies are required by miniSAM. Make sure to activate the correct conda environment when installing the miniSAM Python package, so it is installed in the correct conda environment. Note that when miniSAM is built with __MINISAM_WITH_MULTI_THREADS__ set to ON, the python debugger doesn't work inside a factor class (in my case using VS code). Also, I don't experience much speed improvement when turning mutli-threads on.

## Install SMMPDA Package
This repo comes with a [setup.py](setup.py) file that wraps this project into an installable Python package. Once the project is installed, the library can be imported easily in a Python module. This makes testing more convenient. ([ref](https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944))

To install the smmpda package in _develop_ mode:
```
cd [PATH_TO_REPO]
pip install -e .
```

## Quick Start
Three tasks are performed individually. Remember to activate the correct conda environment.

### 1. Collect data in CARLA with a predefined path.
Launch CARLA, then run:
```
python -O raw_collector.py settings/carlasim.yaml -r
```
This spawns a Mustang (yes, I'm old-school) in the CARLA world with sensors, which wanders around and collects data from the sensors. The ```-O``` flag turns on the optimization mode of python interpretor and turns off debug features. When runing without this flag, some shapes will be drawn in the CARLA world, e.g. lane boundaries points, for debug's purpose. 

The first argument is the configuration YAML file that defines all details regarding CARLA simulation, such as the used map, the weather, how the car is controlled, and sensor configurations. The folder [settings](settings) contains a [carlasim.yaml](settings/carlasim.yaml) that can be used for quick tests. See comments in the files for more information on how to set the parameters. If a sensor noise parameter is set as 0, CARLA simply gives you the ground truth value for the corresponding measurement. In the folder [settings/routes](settings/routes), several pre-defined config files can be found, which only differ in their used waypoints.

The flag ```-r``` turns on the saving of the recorded data specified in the configuration file mentioned above. A __recordings__ folder will be created in the root of this project the first time, under which the recorded data of a simulation will be saved into a folder named by the time the simulation is run. It is recommended to change the folder name to something more meaningful right after saving. Data are stored into 3 files:
1. sensor_data.pkl: Simulated raw sensor data.
2. gt_data.pkl: Ground truth data.
3. carla_recording.log: Data for CARLA replay. Refer to [here](https://carla.readthedocs.io/en/0.9.10/adv_recorder/).

Besides recordings, a copy of the used CARLA simulation configuration file named __config.yaml__ is saved in the folder __settings__ under the same folder for future reference.

### 2. Generate simulated object-level detections
This step doesn't require CARLA. Say you have saved the collected data in __recordings/test__ in the first step, generate simulated detections and the pole map for it by:
```
python -O detection_generator.py recordings/test settings/vision.yaml settings/sim_detection.yaml settings/pole_map.yaml
```
Running it without the ```-O``` flag will show the visualization of the generated pole map in the end. The first argument is the folder of the recording. The file __vision.yaml__ defines parameters regarding vision-based detection algorithms. __sim_detection.yaml__ defines parameters regarding data generated based on GT data, which are more artificial compared to the vision-based part. __pole_map.yaml__ defines parameters that are used for pole map generation. See the comments in them for more information. It took me some time to tune the parameters, but feel free to fine-tune them.

The results will be saved into 2 files under the same folder as the recorded data:
1. detections.pkl: Simulated detections.
2. pole_map.pkl: Generated pole map.

The 3 above-mentioned configuration files are also copied into the __settings__ folder under the same recording folder for future reference.

After generating simulated detections, you can visualize the result by:
```
python detection_viewer.py recordings/test
```

### 3. Run SMMPDA localization on simulated data
Say you have generated simulated detection data in __recordings/test__ in the second step. Launch CARLA (preferably in no rendering mode), then run the following if you have added measurement noise in step 1. and 2..
```
python sliding_window_localization.py recordings/test settings/localization.yaml -s ANY_NAME_YOU_LIKE -ve
```
If you have run step 1. and 2. with simulated noise configured to 0, there is still a way to add post-simulation noise:
```
python sliding_window_localization.py recordings/test settings/localization.yaml -n settings/post_noise.yaml -s ANY_NAME_YOU_LIKE -ve
```
The first argument is the recording folder. __localization.yaml__ defines all parameters regarding SMMPDA localization. The flag ```-n``` turns on post-simulation noise and uses parameter defined in __post_noise.yaml__ to simulate noise. This way you can reuse the same recording to simulate situaions with different noise configurations. Recordings can take up a lot of space. The flag ```-s``` saves the localization results in the folder with a specified name under the folder __results__, which is created the first time localization results are to be saved. The flag ```-ve``` toggles on the visualization of the resulting colored error plots.

In the save folder, 4 files are stored:
1. localizatin.gif: Animation of localization process.
2. results.pkl: Localization results.
3. localizatin.yaml: A copy of SMMPDA localization configuration file for future reference.
4. post_noise.yaml (optional): A copy of post-simulation noise configuraiton file if used.

Note that the first time a CARLA map is used in a localization, a map image is created using pygame for visualization. It is then cached in the folder __cache/map_images__, so it doesn't have to be created again afterwards. 

## Reproduce Localization Tests
To reproduce the tests performed in the thesis, follow these 3 steps:
### 1. Prepare recordings:
Prepare the 5 recordings with the following 5 scenario configuraitons:
1. [urban](settings/routes/town03/urban.yaml)
2. [right_turn](settings/routes/town03/right_turn.yaml)
3. [highway](settings/routes/town04/highway.yaml)
4. [s2](settings/routes/town04/s2.yaml)
5. [s3](settings/routes/town04/s3.yaml)

e.g. ```python -O raw_collector.py settings/routes/town03/urban.yaml -r```

Note: s2 means the ego car on a __straight__ highway starts from the __2nd__ lane and makes a lane change to the right. Similar for s3.

Rename the folders of recordings so you have the same structure in the __recordings__ folder:

```
recordings
??????highway
??????right_turn
??????s2
??????s3
??????urban
```

### 2. Prepare detection data
Use the command as in the section __"Generate simulated object-level detections"__ above to generate detections for each scenario. 

e.g. ```python -O detection_generator.py recordings/highway settings/vision.yaml settings/sim_detection.yaml setting/pole_map.yaml```

Use the default __sim_detection.yaml__ to not introduce simulated errors at this step since post-simulation errors will be added during localization later.

### 3. Run localization tests
First, launch CARLA server (preferably in no rendering mode). A Python script __run_predef_tests.py__ has been prepared that refers to [scenarios.yaml](settings/tests/scenarios.yaml) and automatically run through all configurations in parallel. Simply run:

```python run_predef_tests.py```

The configurations corresponding to the parameters used in the thesis are pre-defined and stored in the folder [settings/tests](settings/tests). The script uses a process pool of 2 by default since using more tends to crash CARLA on my laptop, but try it yourself maybe you have better luck ;). Running the entire test set takes roughly 2 hours on my laptop with i7-10750H. Afterwards, you should be able to find for each recording a folder __results__ containing subfolders named after the configs in __scenarios.yaml__, where the localization results are stored.

To visualize the results, check the folder [helper_scripts](helper_scripts).

## Camera calibration and IPM parameters
The two Jupyter notebooks [front_camera_calibration.ipynb](front_camera_calibration.ipynb) and [ipm_using_deal_vanish_point.ipynb](ipm_using_deal_vanish_point.ipynb) are provided to generate the calibration and IPM parameters of the front-facing camera in an interactive way. The front bumper frame is the reference frame when computing the calibration matrix, so the calbration matrix relates coordinates in the front bumper frame and the image pixels. Both notebooks use the images in the folder [calib_images](calib_images) either for computation or visualization. The results are stored in __calib_data.pkl__ and __ipm_data.pkl__ respectively, which are already part of the repo.

These parameters are essential for detection simulation. If you somehow need to change the configuration of the camera, remember to update these parameters as well. It can be done by reusing the two Jupyter notebooks with a few adjustments.

## Packages
This repo currently contains 4 major packages:
- __carlasim__: Contains modules related to CARLA simulation and data collection.
- __detection__: Contains modules related to detection simulation.
- __localization__: Contains modules implementing SMMPDA localization.
- __model__: Contains CTRV motion model implementation.
