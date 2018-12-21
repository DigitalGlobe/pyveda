def VGG16(**kwargs):
    #assert has_keras, 'To use this model install keras'
    from keras.applications.vgg16 import VGG16
    return VGG16(**kwargs)
