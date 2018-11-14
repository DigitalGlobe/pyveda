Training Data
=============

Training Data Overview
----------------------

The Training Data API stores and fetches training data for modeling purposes. 

Creating New Training Data
--------------------------

All training data sets consist of image/label pairs. An image typically comes from a gbdxtools image class (ie CatalogImages). 
Labels vary depending on the type of data being stored. Training Data in Veda supports 3 types of data:

* classification
    - Used to store classification data in either binary classification or multiclass
    - Classification data are stored using "one hot encoding", so for binary its either [1] or [0]. 
    - For multi-class it'd be: [0, 0, 0, 1, 0] (zero for every class not present and 1 for every class that is present)
* object_detection
    - Object Detection data are arrays of pixel based bounding boxes for each feature present in the corresponding image
    - bboxes are expected to be lists of bboxes: [[minx, miny, maxx, maxy], [minx, miny, maxx, maxy], ...]
    - the bbox values must be in pixel coordinates relative the upper left of the image in the same pair
* segmentation
    - segmentation are expected to be arrays of geojson features in the same projected units as the image.
    - these features are converted to pixel locations when saving into veda and are served back to clients are ndarrays of segmentation data.  

For example we'll create a small set of random binary classification data (values are either 1 or 0). We'll start be using gbdxtools to create an image. 

.. code-block:: python

    from gbdxtools import CatalogImage
    image = CatalogImage('104001001DB7BA00')

This sets up an image for us to use as smaller images in our training data. Next we'll setup the training data set via pyveda:

.. code-block:: python

    from pyveda import VedaCollection

    vc = VedaCollection('Some Fake Data', mltype = "classification", tile_size=[256,256]) 

Now we have an image to pull imagery from and a TrainingSet in which we can store pairs. In the example below we'll set up
the size of the image windows we want to create and a count of pairs. Then we'll access random windows on the image and feed them 
to our TrainingSet. 

.. code-block:: python

    import random

    size_y = 256
    size_x = 256
    count = 100

    for win in image.iterwindows(count=count, window_shape=(size_y, size_x)):
        
        arr = [ random.choice([1, 0]) ]
        td.feed([(win, arr)])

    # verify the count
    print(td.count)
    # outputs: {'train': 100}


Now we have fed 100 pairs of random binary classification data to our TrainingSet. You can see that the count property 
outputs a dictionary `{'train': 100}`. The `feed` method by default puts all data into the `train` group, but you can add pairs 
to groups with any name. For instance lets add another 25 random pairs to the `test` group:

.. code-block:: python

    for win in image.iterwindows(count=25, window_shape=(size_y, size_x)):

        arr = [ random.choice([1, 0]) ]
        td.feed([(win, arr)], group="test")

    # verify the count
    print(td.count)
    # outputs: {'train': 100, 'test': 25}


Saving new data
---------------

Now that we've created a new TrainingSet we'll want to save it into Veda. Saving the data allows Veda to cache all of the imagery 
and provides not only a way to save the data for later use, but also ways to fetch batches of data quickly. Saved sets also
have the benefit of being used by others to train models.

.. code-block:: python

    td.save()

Which will post the image/label pairs and return json: 

.. code-block:: json

    {
        'data': {
            'id': 104,
            'classes': ['class1'],
            'count': {'train': 100, 'test': 25},
            'name': 'Some Fake Data',
            'nclasses': 1,
            'mlType': 'classification',
            'bbox': [-74.03524079, 40.489764494286746, -73.86965922334531, 41.0133572],
            'shape': [8, 256, 256],
            'dtype': 'float32',
            'sensors': ['WV03_VNIR'],
            'source': 'rda',
            'userId': '21668',
            'public': False,
            'created_at': '2018-08-22T16:46:34.838Z',
            'updated_at': '2018-08-22T16:46:34.838Z',
            'percent_cached': 0
        },
        'links': {
            'self': {'href': 'https://veda-api.geobigdata.io/data/104'},
            'update': {'href': 'https://veda-api.geobigdata.io/data/104', 'method': 'PUT'},
            'publish': {'href': 'https://veda-api.geobigdata.io/data/104/publish', 'method': 'PUT'},
            'delete': {'href': 'https://veda-api.geobigdata.io/data/104', 'method': 'DELETE'},
            'create': {'href': 'https://veda-api.geobigdata.io/datapoints', 'method': 'POST'}
        }
    }

As you can see the doc returned is how the TrainingSet is represented on the server and is a self descibing document for updating, deleting, etc. 
Once the TrainingSet is saved to the API Veda immediately begins caching the imagery. Depending on the size of the data this can take several minutes, 
but as the caching progresses the property `percent_cached` is updated. 


Publishing
----------

Once we've saved a dataset we can `publish` it for others to use. This action makes the entire set of data available for other users. 

.. code-block:: python
      
      td.publish()

Unpublishing 
------------

If you published data that you want to revoke access to anyone but yourself you can call `unpublish`:

.. code-block:: python

      td.unpublish()


Searching for Training Data
---------------------------

Now that we've saved some data we can search for it. The search method allows us to find datasets of interest and allows us to find datasets
matching certain search criteria like spatial bounding boxes, names, etc. This method will return all the matching TrainingSets as an array of json docs.  

.. code-block:: python

        from pyveda import search 

	for s in search():
            print(s.id, s.name, s.percent_cached)


Using Training Data 
-------------------

To use a TrainingSet found via `search` we offer two methods: `TrainingSet.from_doc` and `TrainingSet.from_id`. Both of these methods do the same thing, that is 
they create a TrainingSet from which data can be fetched and used to train a model. 


.. code-block:: python

        # Using from_id
        td = TrainingSet.from_id(104)

        # Using from_doc
        sets = search()
        td = TrainingSet.from_doc(sets[0])

Both achieve the same result, that is an instance of TrainingSet that can be used for training. 

Working with data
-----------------

Now that we're ready to access some data for training we need to learn about the various ways for accessing pairs from the TrainingSet. 
Getting data can be done in a few ways, either sequentially or in batches. 

To explore the structure of one data pair in the set we can simple print the first one. You can think of a TrainingSet as deferred 
access to the raw image (images can be heavy), but we can fetch it on demand as a way to explore the data. 

Note: The thing to understand here is that pairs inside sets have different properties for access the raw image and label data. The `.image` property
returns a dask (deferred numpy array) to the image data for each point. This dask will only return the raw pixel values of the image when you call `image.compute()`. 
The `.y` property is the label data for the given pair.  

.. code-block:: python

          # get the first 5 pairs
          pairs = td[0:5]

          # view whats stored as json
          print(pairs[0].data)
          
          # get the image data
          print(pairs[0].image.compute())

          # get the label data for the first point
          print(pairs[0].y)

Using this access pattern is generally useful for doing small explorations on TrainingSets. In order to actually with data we need to 
fetch more it, but this can also be dangerous due to the size of some TrainingSets. We need to be careful about not overloading the memory of our process
by fetching too many images into memory at most. We'll see how to protect ourselves from that soon. 

Fetches batches
---------------

A batch of data is a useful way to start training models. Batches return image/labels as numpy arrays that are ready for training:
This writes the data into an HDF5 file that acts as persistent cache of data. This is useful on many fronts but most importantly it saves us from using too much memory.
Batches return instances of `ImageTrainer`, which can be used to iterate over data and train models.  

.. code-block:: python

          # get the first 5 pairs
          batch = td.batch(32)
          print(len(batch.train))

Using batches
-------------

.. code-block:: python

    for x,y in b.train[:5]:
        print(x,y)

Releases
--------

Coming Soon...
