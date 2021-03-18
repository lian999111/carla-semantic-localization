# SMMPDA

This repo implements Semantic Max-mixture Probabilistic Data Association localization using the miniSAM library. Simulations are conducted with CARLA.

## Environment Setup
Using miniconda is recommended beause this repo is developed this way. The repo comes with an [environment.yml](environment.yml) file that facilitates setting up the environment.

To create a conda environment with the default name "__smmpda__":
```
conda env create -f environment.yml
```
Refer to [Creating an environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more info.

## Dependencies
This project depends on CARLA and miniSAM, which need to be installed manually in addition to the conda environment. Follow their official guides for proper installation.

[CARLA](http://carla.org/)  
The version __0.9.10__ is used throughout this repo. The corresponding .egg file has be included in the root directory of this repo. To upgrade, replace the .egg file with a newer version. CARLA suggests two ways to import PythonAPIs as written in [here](https://carla.readthedocs.io/en/latest/build_system/). The first inelegant method by adding the .egg file to the system path is used because _easy_install_ seems to be deprecated ([ref1](https://setuptools.readthedocs.io/en/latest/deprecated/easy_install.html), [ref2](https://workaround.org/easy-install-debian)).

[miniSAM](https://minisam.readthedocs.io/index.html)  
Follow the instructions in [C++ Compile and Installation](https://minisam.readthedocs.io/install.html) and [Python Installation](https://minisam.readthedocs.io/install_python.html) to install the miniSAM library. Some additional dependencies are required by miniSAM. Make sure to activate the correct conda environment when installing the miniSAM Python package, so it is installed in the correct conda environment.

## Install SMMPDA package
This repo comes with a [setup.py](setup.py) file that wraps this project into an installable Python package. Once the project is install, the library can be impored by any scripts within this project. This makes testing more convenient. ([ref](https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944))

To install the smmpda package in _develop_ mode:
```
cd [PATH_TO_REPO]
pip install -e .
```
