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

The bounds.geojson file contains a geojson feature describing the bounds of the release data.


Releases for Importing
-------------------------

The release format can also be used for imported existing data in Veda using the Bulk Loading API. The `meta.json` and `bounds.geojson` files are not needed for importing. The only requirement is that the data are stored in a tar.gz file, and follow the image and tile structure shown above. Each tile and corresponding json file needs to have the same base name and the extensions `.tif` and `.json`.

Images must in tif format. They do not need to be geotiff files. The image's georeferencing information is taken from the matching json file. 

The json file is a single geojson feature consisting of a `Polygon `describing the image bounds. The label for the image is stored in the feature's `Properties`. The label format is the same as described in the :ref:`Label Formats` section.

A simplified example of the contents of the json file for classified images is:

..code-block:: python

    >>> cat labels/tile_0.json

    {
    "geometry": {
        "type": "Polygon", 
        "coordinates": [[...]]
    }, 
    "properties": {
        "label": {
            "clouds": 1,
            "building": 0,
            "car": 1
        }
    }
    }

Images labelled for segmentation or object detection will have a json file like:

..code-block:: python

    >>> cat labels/tile_0.json

    {
    "geometry": {
        "type": "Polygon", 
        "coordinates": [[...]]
    }, 
    "properties": {
        "label": {
            "building": [
                {"type": "Polygon", "coordinates": [[[]]]},
                {"type": "Polygon", "coordinates": [[[]]]}
            ],
            "car": [
                {"type": "Polygon", "coordinates": [[[]]]}
            ]
        }
    }
    }

Images that do not have georeferencing are also supported. If the geometry field is left empty Veda will treat the image as having a pixel-based coordinate system. Label geometries, if present, are described in units of pixels using the top left of the image as the origin. Data points without spatial information can not be found with spatial searches or filters.

..code-block:: python

    >>> cat labels/tile_0.json

    {
    "geometry": {}, 
    "properties": {
        "label": {
            "clouds": 1,
            "building": 0,
            "car": 1
        }
    }
    }

Once the images and labels are collected in the correct formats and directory structure, they need to be compressed as a `tar.gz` file and placed in a publicly available accessible S3 bucket.

The data can be imported using the :meth:`pyveda.main.create_from_tarball` method, see the :ref:`Creating a Collection using the Bulk Import API` section. 
