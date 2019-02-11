Model Training 
================

             

.. code-block:: python
                
    from pyveda.pretrainedmodels.keras.classification.resnet50 import ResNet50

    model = ResNet50()
    ds = pv.open(filename='buildings.vdb')

    batch_size = 16
    tester = ds.test.batch_generator(batch_size) 
    trainer = ds.train.batch_generator(batch_size)

    # Train the model...
    epochs = 1
    model.fit_generator(trainer, steps_per_epoch=len(trainer), epochs=epochs, validation_data=tester)


              

            

