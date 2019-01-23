Getting started
===============

**Veda is a library of object knowledge for machine learning**.

Veda stores organized collections of images and their labels. It is integrated with the GBDX library of imagery and can access almost 20 petabytes of satellite images to use for generating image chips. Data sets and models can be published and shared with others.

**Veda is spatially aware.**

Veda "speaks" geojson and can convert spatial data into machine learning data and back. Use spatially-targeted training sets to develop location-specific models.

**Pyveda is the bridge between Veda and machine learning platforms**

Pyveda is a Python access library that lets you generate and manage machine learning data. When you're ready to train or test your model with that data, pyveda supplies the data to your algorithms in efficient formats that are ready to analyze. When your model is trained, you can also use pyveda to upload the model to Veda.

**Veda does not train or run models - your choice of library and platform is up to you.**

Getting access to Veda
-----------------------

Access to Veda requires GBDX credentials. You can sign up for a GBDX account at https://gbdx.geobigdata.io.

Your GBDX credentials are found under your account profile.

PyVeda's authorization tools expect a config file to exist at ~/.gbdx-config with your credentials.
(See formatting for this file here:  https://github.com/tdg-platform/gbdx-auth#ini-file.)

For questions or troubleshooting email GBDX-Support@digitalglobe.com.

Installing pyveda
-------------------

.. code-block:: python

    pip install pyveda
