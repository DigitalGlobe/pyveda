Models and Predictions
=============

Model Overview
----------------------

While Veda and Pyveda do not directly train models, they can store and share models. They can then take a stored model and deploy it to Amazon SageMaker to run predictions on images. They can also run predictions over large areas at scale. The results of the analysis can then be fed back into training data.

Models in Veda
----------------

Veda's concept of a model is an object that stores a trained model. This includes the model's structure and weights, as well as metadata about the model. Metadata includes important information such as the size and bands of image tiles the model can accept, or the machine learning platform it was developed in. If the model has been deployed to SageMaker, it also stores information about the endpoint. Any data sets generated when running bulk prediction on areas are stored in the model object too.

Veda supports the following machine learning frameworks:

* Keras
* PyTorch
* TensorFlow
* ONNX

Veda can store the models and weights from other frameworks but will not be able to deploy them to SageMaker. You can however share the model and weights so others can download your model.


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

To load a Model object, you can use the ``from_id()`` method and the model's unique ID.

.. code-block:: python

    m = Model.from_id('0c0147f6-de9a-470d-9746-6f44af7e4019')

To update an existing Model, object, pass a new metadata object to the ``update`` method:

.. code-block:: python

    metadata = {...}
    m.update(metadata)

To remove a model, call the ``remove()`` method. Note that there is no confirmation, and the delete operation cannot be undone.

.. code-block:: python
    m.remove()

Publishing a model makes it accessible to other users. They will be able to find the model in searches, and run predictions against the model. You can also unpublish a model to hide it from public searches. 

.. code-block:: python
    m.publish()
    m.unpublish()

Pyveda can download the model files locally. A directory named with the model ID is created and the archive saved as ``model.tar.gz``. To save the model archive to a different location, provide the path using the ``path`` keyword.

.. code-block:: python

    m.download()
    m.download(path='./my_models/cool_model')

The archive will contain the files that were originally stored, so it may contain the model and weights in a single file or two separate files.
    

Creating Models
-----------------

To create a model, Veda needs to know the following information about the model:

* a name for the model
* sources for the model file and weights
* the framework used the generate the model
* the training data used to train the model, if supplied by Veda
* the required shape of input images
* the label format

Since many of these items are derived from the training data, they can be inferred from the dataset used to train the model via the ``training_set`` keyword.

.. code-block:: python

    vc = pv.from_id('db3c619b-8774-4051-a330-a21771822586')

    m = Model('XView Burkina Faso Model 2', 
                archive='./model_test.tar.gz',
                library="keras",               
                training_set=vc,
                )  
    m.save()

If the model was trained from data independent of Veda, instead of ``training_set`` you must provide the ``imshape`` and ``mltype`` arguments. See the dataset section for more information about these fields.

.. code-block:: python

    m = Model('XView Burkina Faso Model 2', 
                archive='./model_test.tar.gz',
                library="keras",               
                imshape=(256,256,1),
                mltype="segmentation"
                )  
    m.save()
    
The model file and weights can be supplied already compressed in a tar archive matching a specific format. The model file must be named ``model.json`` and the weights ``model_weights.h5``. If using a combined model and weight H5 file, the file must be named ``model.h5``.

.. code-block:: python

    m = Model('XView Burkina Faso Model 2', 
                archive='./model_test.tar.gz',
                library="keras",               
                training_set=vc,
                )  
    m.save()

The model and weight files can also be passed separately, and pyveda will create a correctly formatted archive for you. The individual files can have any name.

.. code-block:: python

    m = Model('XView Burkina Faso Model 2', 
                model_path='model_v2.h5',
                weights_path='model_weights_v2.h5', 
                library="keras",               
                training_set=vc,
                )  
    m.save()

To specify the model framework, the ``library`` argument can be any value. Only Keras, TensorFlow, PyTorch, and ONNX models can be deployed to SageMaker. These should be passed in lower case - ``keras``, ``tensorflow``, ``pytorch``, or ``onnx``.

As shown in the examples, you must call ``save()`` on the model after creating it.

Deploying Models to SageMaker
--------------------------------

The model can be deployed to Amazon SageMaker with one command:

.. code-block:: python

    m.deploy() 

Veda will upload the model files, create the correct Docker container for the model's machine learning framework, and create the invocation endpoint for the container.

Deployment information can be found in the following:

TBD

.. code-block:: python

    m.refresh() # get the current information about the model
    m.deployed # boolean of whether the model is completely deployed
    m.endpoint # Name of the endpoint

Once the deployment is complete it can be accessed through the methods provided by Amazon as well as through the Veda API:

TBD

If you update the model, TBD

Running Predictions
---------------------

To use the deployed model to predict labels for an image, you can use the ``predict()`` method:

TBD

.. code-block:: python

    image = './images/image1.png' 
    labels = m.predict(image)

TBD the predict method can accept the following values for the image:

* a path to a local file
* a publicly available S3 location
* a NumPy array (bands first), i.e [3,256,256])
* any ``gbdxtools`` image object, like CatalogImage

While you can run predictions on single images,  Veda can also run predictions over whole areas using the ``bulk_predict`` method. The image source can be any ``gbdxtools` image object. This opens up the source for imagery to be any image type provided by RDA, which includes DigitalGlobe satellites as well as public satellite imagery from sources like Sentinel.

The prediction service will automatically tile the source imagery, run a prediction on the image tiles, and save all the tiles with their new labels in a PredictionSet. Like DataSets, a PredictionSet is a collection of image and label pairs.

The function takes a reference to a ``gbdxtools`` image object, and a bounding box of the area to analyze. Any image object can be used as long as it's in a format the model expects - i.e. they should have the same number of bands, and the bands should represent similar regions of the spectrum.

.. code-block:: python

    from pyveda.veda.rda import MLImage

    img = MLImage('1030010038CD4D00')
    from shapely.geometry import box, shape, mapping

    aoi = img.randwindow((1000,1000)).bounds

    m.bulk_predict(aoi, img)

To check the progress of the prediction job, TBD



Viewing Predictions
---------------------

The predictions run from a model can be found at:

TODO

To search for predictions, use:

TBD

This is useful in the event you have run predictions against a public model that was later unpublished.

Predictions can also be inspected via the FeatureServer endpoint. This can be added to any map that supports the vector tile format.

TBD

The features in a PredictionSet can also be downloaded as a GeoJSON file:

TBD



Turning Predictions into Data
--------------------------------

PredictionSets are also similar to DataSets in that they can be cleaned up using the same validation and verification tools. Inside Jupyter Notebooks, you can use the Labelizer tool. 

TBD

You can also clean up PredictionSets using the tools in Information Product Hub.

To promote all the datapoints inside of a Predictionset into a DataSet so that it can be used for training, use:

TBD

To move an individual datapoint from a PredictionSet to a DataSet, use the ``move`` and ``copy`` methods on the datapoint:

TBD