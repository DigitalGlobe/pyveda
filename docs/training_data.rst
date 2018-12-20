Training Data
=============

.. note:: This tutorial is a work in progress

Training Data Overview
----------------------

The Training Data API stores and fetches training data for modeling purposes. 

Creating New Training Data
--------------------------

Veda is a repository for machine learning data. Its focus is on generating and storing image and label pairs of spatial data. It can read spatial data and then generate the images from DigitalGlobe's catalog of satellite imagery.

To generate data Veda needs to know three things:

- WHERE is the feature
- WHAT is the feature
- WHICH image to use

The WHERE and WHAT of the features are inputted using the GeoJson format. The image can be any gbdxtools Image object from the DigitalGlobe catalog. Veda will handle generating the appropriate tiles to cover the input features. The same basic concept applies to the three machine learning types that Veda supports: classification, object detection, and classification.

All training data sets consist of image/label pairs. The training images are fixed-sized chips extracted from the source image. The labels vary depending on the type of data being stored:

* Classification
    - Classification describes whether an object class exists in the image.
    - Multiple classes are supported.
    - Classification data are stored using "one hot encoding", so for a single class the label is either [1] or [0]. 
    - For multi-class the labels are lists: [0, 0, 0, 1, 0] (zero for every class not present and 1 for every class that is present).
* Object Detection
    - Object detection describes where in an image objects occur.
    - Object Detection data are arrays of pixel based bounding boxes for each feature.
    - Labels are stored as lists of bounding boxes: [[minx, miny, maxx, maxy], [minx, miny, maxx, maxy], ...]
    - The bounding box values must be in pixel coordinates relative the upper left of the image in the same pair
* Segmentation
    - Segmentation also describes where in an image objects or classes occur but uses freeform boundaries instead of boxes.
    - Segmentation data are expected to be arrays of geojson features in the same projected units as the image.
    - These features are converted to pixel locations when saving into Veda and are served back to clients are ndarrays of segmentation data.  

Sample Data
--------------

The PyVeda git repository includes a sample geojson file to get started with. The file ``sample_data/turbines.geojson`` contains bounding boxes of wind turbines in the Alta Wind Energy Center, located outside of the Mojave Desert in California. Each feature has a property called ``label`` with the value ``turbine``.

For a corresponding image to generate the chips from we'll use catalog ID '123456788' from DigitalGlobe's Worldview 2 satellite. To search for other images covering this area see the "Searching for images" section below.



Creating a VedaCollection
-----------------------------

A VedaCollection stores machine learning data in the Veda system. As mentioned above, it needs the inputs of geojson features and an Image object. For the geojson we'll use the sample wind turbine data. We'll also need a gbdxtools Image object. This could be a CatalogImage or a complicated RDA output; gbdxtools gives you almost unlimited processing options for generating images. The images could be a custom mix of bands or have pansharpening applied, for instance. 

For most cases we would like imagery that just has the three visible RGB bands, has been pansharpened for maximum detail, atmospheric effects removed, and dynamic range adjusted to an optimum range. These processing options are all included in PyVeda's MLImage class. Only a Catalog ID is needed to create an MLImage object.

First, let's get set up:

.. code-block:: python

    from pyveda import VedaCollection, DataSet, MLImage

    catID = '123456789'
    geojson = 'path/to/data/turbines.geojson'

Next, let's create our VedaCollection. It requires some basic parameters to get started:

* A name for the collection
* The type of machine learning algorithm, in this case image classification
* A tile size to use for the images.

.. code-block:: python

    vc = VedaCollection('Wind Turbines', mlType="classification", tilesize=[256,256]

This sets up the VedaCollection but it will not be created on the server until some data is loaded. So lets load some data:

.. code-block:: python

    image = MLImage(catID)
    vc.load(geojson, image)


It will take about 5 minutes for Veda to generate all the training data. You can check `vc.status` to track its progress. 

The end result will be collection of 256x256 pixel image tiles extracted from catalog ID 123456. Each image will have a corresponding label of the class `turbine`. Because this analysis is classification, each label will be `1`.

(more stuff about VCs)

Working with DataSets
------------------------

A DataSet is a local copy of training data. You can specify the number of samples to load. DataSets also support internal partitions for training, validation, and testing data, so we can specify the percentage of points to put into each group.

.. code-block:: python
    ds = vc.store('path/to/my_test.h5', partition=[70,10,20], size=100)

This will download 100 training points to a local file `my_test.h5`. 70 points will be assigned to the training group, 10 to the validation group, and 20 to the testing group. The partition is optional, and if not specified all the points are put into the training group.

We'll use the training data to build a simple FOO model:

.. code-block:: python

    from FOO import FOOMODEL

    model = FOOMODEL()
    model.train(ds.train.images, ds.train.labels)
    model.score(ds.test.images, ds.test.labels)





Releases
--------

Coming Soon...
