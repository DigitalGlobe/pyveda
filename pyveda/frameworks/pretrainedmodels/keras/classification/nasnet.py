def NASNetLarge(**kwargs):
    #assert has_keras, 'To use this model install keras'
    from keras.applications.nasnet import NASNetLarge
    return NASNetLarge(**kwargs)
