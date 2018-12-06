def ResNet50(**kwargs):
    #assert has_keras, 'To use this model install keras'
    from keras.applications.resnet50 import ResNet50
    return ResNet50(**kwargs)
