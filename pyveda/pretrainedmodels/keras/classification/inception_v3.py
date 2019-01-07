def InceptionV3(**kwargs):
    #assert has_keras, 'To use this model install keras'
    from keras.applications.inception_v3 import InceptionV3
    return InceptionV3(**kwargs)
