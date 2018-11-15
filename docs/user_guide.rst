Getting started
===============

Veda
------

Veda is DigitalGlobe's platform for machine learning with satellite data. It provides tools for generating, organizing, and publishing training data and models.

PyVeda is a Python module to interface with the Veda platform and integrate its training data into Python machine learning libraries such as PyTorch.

Getting access to GBDX
-----------------------

Access to Veda requires GBDX credentials. You can sign up for a GBDX account at https://gbdx.geobigdata.io.

Your GBDX credentials are found under your account profile.

PyVeda's authorization tools expect a config file to exist at ~/.gbdx-config with your credentials.
(See formatting for this file here:  https://github.com/tdg-platform/gbdx-auth#ini-file.)

For questions or troubleshooting email GBDX-Support@digitalglobe.com.

Overall Concepts 
---------------------------

Veda is designed to store training data and models related to Machine Learning from satellite imagery. 

The general concept is to use Veda to either find/search for existing training datasets or 
create new training datasets. These training sets consist of image/label pairs for that are stored inside veda and made available via its API. Additionally Veda supports the same pattern to save and retrieve trained models.  

The goal is to provide a ML system for making access to training data and models as easy as possible. Additionally its goals are to provide ways for users to enhance models and training data through model iteration (verification & validation of model results).

Veda is not a machine learning library. It provides the infrastructure for other machine learning system: storage, discovery and access.  

