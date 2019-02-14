import sys
import os
from setuptools import setup

req_path = os.path.join(
    os.path.abspath(
        os.path.dirname(__file__)),
    "requirements.txt")
with open(req_path) as f:
    reqs = f.read().splitlines()

if sys.version_info[0:2] >= (3, 4):
    reqs.append("aiohttp")

setup(
    name='pyveda',
    version='0.0.1',
    author='DigitalGlobe',
    author_email='',
    description='Python API for interfacing with veda',
    url='https://github.com/DigitalGlobe/pyveda',
    license='MIT',
    packages=[
        'pyveda',
        "pyveda.fetch",
        "pyveda.fetch.compat",
        "pyveda.fetch.aiohttp",
        "pyveda.fetch.diagnostics"],
    zip_safe=False,
    install_requires=reqs)
