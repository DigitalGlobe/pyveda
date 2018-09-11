Getting started
===============

Getting access to GBDX
-----------------------

All operations on GBDX require credentials. You can sign up for a GBDX account at https://gbdx.geobigdata.io.
Your GBDX credentials are found under your account profile.

All auth is performed via gbdxtools' auth tools and gbdxtools expects a config file to exist at ~/.gbdx-config with your credentials.
(See formatting for this file here:  https://github.com/tdg-platform/gbdx-auth#ini-file.)

Instantiating an Interface object automatically logs you in:

.. code-block:: pycon

   >>> from gbdxtools import Interface
   >>> gbdx = Interface()

For questions or troubleshooting email GBDX-Support@digitalglobe.com.

Overall Concepts 
---------------------------

Veda is designed to store training data and models related to Machine Learning from satellite imagery. 
The general concept is to first either use Veda to either find/search for existing training datasets or 
be used to save new training datasets. These training sets consist of image/label pairs for that are stored
inside veda and made available via its API. Additionally Veda supports the same pattern for save/fetching trained models.  

The goal is provide a ML system for making access to data and models as easy as possible. Additionally its goals are 
to provide ways for users to enhance models and training data through model iteration (verification & validation of model results).

Veda does NOT support training your models. It provides storage and access only.  

