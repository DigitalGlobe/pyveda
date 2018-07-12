import sys
import os
from setuptools import setup

reqs = ["gbdxtools", "h5py", "tables"]
if sys.version_info[0:2] >= (3,4):
    reqs.append("aiohttp")

setup(name='pyveda',
      version='0.0.1',
      description='Python API for interfacing with veda',
      url='https://github.com/DigitalGlobe/pyveda',
      license='MIT',
      packages=['pyveda', "pyveda.fetch", "pyveda.fetch.compat", "pyveda.fetch.aiohttp", "pyveda.fetch.diagnostics"],
      zip_safe=False,
      install_requires=reqs
      )
