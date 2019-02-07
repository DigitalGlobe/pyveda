Creating Data
=============

.. note:: This tutorial is a work in progress. While Veda is set in its functionality, the pyveda 
interface is steadily evolving.

The previous section looked at how to access data from Veda. Now we'll look at how to create your own training data.

Creating New Training Data
--------------------------

Veda is a repository for machine learning data.  It can also generate data. It can read spatial data and then generate the images from DigitalGlobe's catalog of satellite imagery. It can also use the spatial data's feature properties to generate labels.

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

See the :ref:`Label Formats` for more information and examples.

.. warning:: This is not totally set up to use like a tutorial yet, sorry!

Sample Data
--------------

The PyVeda git repository includes a sample geojson file to get started with. The file ``sample_data/turbines.geojson`` contains bounding boxes of wind turbines in the Alta Wind Energy Center, located outside of the Mojave Desert in California. Each feature has a property called ``label`` with the value ``turbine``.

For a corresponding image to generate the chips from we'll use catalog ID '123456788' from DigitalGlobe's Worldview 2 satellite. To search for other images covering this area see the "Searching for images" section below.


Creating a Collection from Spatial Data
---------------------------------------------

A VedaCollection stores machine learning data in the Veda system. As mentioned above, it needs the inputs of geojson features and an Image object. For the geojson we'll use the sample wind turbine data. We'll also need a gbdxtools Image object. This could be a CatalogImage or a complicated RDA output; gbdxtools gives you almost unlimited processing options for generating images. The images could be a custom mix of bands or have pansharpening applied, for instance. 

For most cases we would like imagery that just has the three visible RGB bands, has been pansharpened for maximum detail, atmospheric effects removed, and dynamic range adjusted to an optimum range. These processing options are all included in PyVeda's ``MLImage`` class. Only a Catalog ID is needed to create an MLImage object.

First, let's get set up:

.. code-block:: python

    import pyveda as pv
    from pyveda.rda import MLImage

    catID = '123456789'
    geojson = 'path/to/data/turbines.geojson'

Next, let's create our VedaCollection. It requires some basic parameters to get started:

* The geojson source to use
* The gbdxtools image object to use for the source imagery
* A name for the collection.

There are other parameters to set like the size of the images, but to start we can use the default settings.

.. code-block:: python

    name = 'Wind Turbines' # give this a more distinctive name
    image = MLImage(catID)
    vc = pv.create_from_geojson(gojson, image, name)
    
It will take about 5 minutes for Veda to generate all the training data. You can check ``vc.status`` to track its progress. 

The end result will be collection of 256x256 pixel image tiles extracted from catalog ID 123456. Each image will have a corresponding label of the class `turbine`. Because this analysis is classification (the default type), each label will be `1`.

(more stuff about creating VCs)


Creating a Collection using the Bulk Import API
---------------------------------------------------

If you would like to import existing image and label data, you can have Veda download and process a compressed archive of the data using the Bulk Import API:

.. code-block:: python

    pv.create_from_tarball('s3://path/to/tarball', 'Collection Name')

The compressed archive needs to follow the pyveda Release format, as described in the :ref:`Releases for Importing` section. It also has to be stored in Amazon S3.

 Running this command will create a new collection in Veda and it can be accessed with the standard access methods of :meth:`pyveda.main.open` and :meth:`pyveda.main.store`.

Adding Data to Existing Collections
-------------------------------------


Creating Samples from Scratch
--------------------------------