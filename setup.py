import sys
import os
from setuptools import setup, find_packages

req_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "requirements.txt")
with open(req_path) as f:
    reqs = f.read().splitlines()

setup(name='pyveda',
      version='0.0.7',
      author='DigitalGlobe',
      author_email='',
      description='Python API for interfacing with veda',
      url='https://github.com/DigitalGlobe/pyveda',
      license='MIT',
      packages=find_packages(exclude=['docs','tests']),
      zip_safe=False,
      install_requires=reqs
      )
