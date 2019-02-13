Releases
===========

Releases are compressed files storing metadata files and two directories, one for images and one for labels. ``meta.json`` contains information about the training set. ``bounds.geojson`` is a geojson FeatureSet containing the bounds of all the image chips. Images and labels have matching names based on their ID, so a typical file stucture looks like:

::

    ts_104-1.0.1.tar.gz

    ts_104-1.0.1 
    |    meta.json
    |    bounds.geojson
    |
    └--  images
    |       1.tif
    |       2.tif
    |       3.tif
    └--  labels
            1.json
            2.json
            3.json
        
Images are saved as geotiff.

The meta.json file includes:

.. code-block:: javascript

    {'bbox': [<coordinate list>],
    'classes': <class data, varies>,
    'mlType': <classification type, str>,
    'name': <name, str>,
    'nclasses': <number of classes, int>,
    'public': <bool>,
    'source': <image source, str>,
    'version': <release version, str>,
    's3_location': <S3 location, str>,
    'source_url': <url of parent dataset in Veda, str>}


Releases for Importing
-------------------------
If you are converting existing data to this format for the purpose of uploading, only the following files are needed:

tbd