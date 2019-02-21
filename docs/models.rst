Models and Predictions
=============

Model Overview
----------------------

While Veda and Pyveda do not directly train models, they can store and share models. They can take a stored model and deploy it to Amazon SageMaker to run predictions on image. They can also run predictions over large images at scale. The results of the analysis can then be fed back into training data.

Models in Veda
----------------

Veda's concept of a model is an object that stores a trained model. This include's the model's structure and weight, as well as metadata about the model, such as the size and bands of image tiles it can accept, or the machine learning platform it runs in. If the model has been deployed to SageMaker, it also stores information about the endpoint. The Model object also stores any PredictionSets generated when running bulk prediction on images.

Searching for Models
----------------------

Similar to DataSets, Models are searchable. Model searches only return metadata about the model.

.. code-block:: python

    import pyveda as pv
    pv.config.set_local()

    from pyveda.models import Model, search

    models = search()

    for m in models:
        print(m.name, m.id)
        print('\t deployed', m.deployed)

To instantiate a Model object, you can use the ``from_id`` method and the model's unique ID.

.. code-block:: python

    m = Model.from_id('0c0147f6-de9a-470d-9746-6f44af7e4019')

To update an existing Model, object, pass a new metadata object to the ``update`` method:

.. code-block:: python

    metadata = {...}
    m.update(metadata)

To remove a model, call the `remove` method. Note that there is no confirmation, and the delete operation cannot be undone.

.. code-block:: python
    m.remove()

Publishing a model makes it accessible to other users. They will be able to find the model in searches, and run predictions against the model.

.. code-block:: python
    m.publish()
    m.unpublish()

Pyveda can download the model files locally. A directory named with the model ID is created and the archive saved as ``model.tar.gz``. To save the model archive to a different location, provide the path using the ``path`` keyword.

.. code-block:: python

    m.download()
    m.download(path='./my_models/cool_model')

TBD In the archive the model file is named ``model.h5`` and the weights ``model_weights.h5``.
    

Uploading Models
-----------------

TBD (loose around the model file name/format)
In the archive the model file must be named ``model.json`` and the weights ``model_weights.h5``.

.. code-block:: python

    vc = pv.from_id('db3c619b-8774-4051-a330-a21771822586')

    m = Model('XView Burkina Faso Model 2', 
                archive='./model_test.tar.gz',
                library="keras",               
                training_set=vc,
                imshape=(256,256,1),
                mltype="segmentation"
                )  

    m.save()

The model and weight files can also be passed separately, and pyveda will create a correctly formatted archive for you. The individual files can have any name.

.. code-block:: python

    model = Model('XView Burkina Faso Model 2', 
                model_path='model_v2.h5',
                weights_path='model_weights_v2.h5', 
                library="keras",               
                training_set=vc,
                imshape=(256,256,1),
                mltype="segmentation"
                )  

Deploying Models to SageMaker
--------------------------------

The model can be deployed to Amazon SageMaker with one command:

.. code-block:: python

    m.deploy() 

Veda will upload the model files, create the correct Docker container for the model's machine learning framework, and create the invocation endpoint for the container.

The endpoint name can be found TODO. Once the deployment is complete it can be accessed through the methods provided by Amazon as well as through the Veda API:

TBD

Running Predictions
---------------------

While you can run predictions on single images using any of the SageMaker invocation methods, Veda can run predictions over whole areas. The image source can be any image source provided by RDA, which includes DigitalGlobe satellites as well as public satellite imagery from sources like Sentinel.

The prediction service will automatically tile the source imagery, run a prediction on the image tiles, and save all the tile with their new labels in a PredicionSet.

The function takes a reference to a ``gbdxtools`` image object, and a bounding box of the Area of Interest. Any image object can be used as long as it's in a format the model expects - i.e. they should have the same number of bands, and the bands should represent similar bandwidths. 

.. code-block:: python

    from pyveda.veda.rda import MLImage

    img = MLImage('1030010038CD4D00')
    from shapely.geometry import box, shape, mapping

    aoi = img.randwindow((1000,1000)).bounds

    m.predict(aoi, img)

To check the progress of the prediction job, TBD



Viewing Predictions
---------------------

The predictions run from a model can be found at:

TODO

To search for predictions, use:

TBD

This is useful in the event you have run predictions against a public model that was later unpublished.

Predictions can also be inspected via the FeatureServer endpoint:

TBD

The features in a PredictionSet can also be downloaded as a GeoJSON file:

TBD

This can be added to any map that supports the vector tile format.


Turning Predictions into Data
--------------------------------

PredictionSets are also similar to DataSets in that they can be cleaned up using the same validation and verification tools. Inside Jupyter Notebooks, you can use the Labelizer tool. 

TBD

You can also clean up PredictionSets using the tools in Information Product Hub.

To promote all the datapoints inside of a Predictionset into a DataSet so that it can be used for training, use:

TBD

To move an individual datapoint from a PredictionSet to a DataSet, use the ``move`` and ``copy`` methods on the datapoint:

TBD