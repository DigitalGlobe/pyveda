def VGG19(**kwargs):
    #assert has_keras, 'To use this model install keras'
    from keras.applications.vgg19 import VGG19
    return VGG19(**kwargs)
