Models
=========

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

.. code-block:: python

    model.save('model.h5')
    model.save_weights('model_weights.h5')
    !tar -cvzf model_test.tar.gz model.h5 model_weights.h5