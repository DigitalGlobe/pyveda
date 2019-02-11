Models
=========


Saving a Model 
----------------

.. code-block:: ipython3

    my_model.save('model.h5')
    my_model.save_weights('model_weights.h5')
    !tar -cvzf model_test.tar.gz model.h5 model_weights.h5

Uploading a Model to Veda
---------------------------

.. code-block:: python

    from pyveda.models import Model
    import pyveda as pv

    pv.config.set_dev()

    vc = pv.from_id('db3c619b-8774-4051-a330-a21771822586')

    model_path = './model.tar.gz'

    model = Model('XView Burkina Faso Model', 
                model_path, 
                library="keras", 
                bounds=vc.bounds, 
                mltype="object_detection", 
                shape=vc.imshape, 
                dtype=vc.dtype,
                training_set=vc.id)

    model.save()

Deploying a Model
------------------

.. code-block:: python

    m = Model.from_id('930638be-a247-423a-bdca-987eb19026689')
    m.deploy()

Searching for Models
-----------------------

.. code-block:: python

    from pyveda.models import search as model_search

    for m in model_search():
        print(m.name, m.id)
        print('\t location:', m.location)
        print('\t lib:', m.library)
        print('\t type:', m.mltype)
        print('\t deployed', m.deployed)
        print('\t public:', m.public, '\n')


Downloading a Model
---------------------

.. code-block:: python

    m = Model.from_id('930638be-a247-423a-bdca-987eb19026689')
    m.download(path='my_local_model.tar.gz')

    !ls my_local_model2.tar.gz


Predicting with Veda Models
-----------------------------

.. code-block:: python

    image = MLImage(...)
    AOI = [...]
    model.predict(image, aoi)


    model.status