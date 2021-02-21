"""Setup for installing the project locally.

Once the project is install, the library can be impored by any scripts within this project.
This makes testing more convenients.

Ref: https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944"""

from setuptools import setup, find_packages

setup(name='smmpda', version='1.0', packages=find_packages())
