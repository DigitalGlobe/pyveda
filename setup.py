from setuptools import setup

setup(name='pyveda',
      version='0.0.1',
      description='Python API for interfacing with veda',
      url='https://github.com/DigitalGlobe/pyveda',
      license='MIT',
      packages=['pyveda'],
      zip_safe=False,
      install_requires=[
          "gbdxtools",
          "h5py",
          "tables"
        ]
      )
