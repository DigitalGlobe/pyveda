def InceptionV3(**kwargs):
    import torchvision.models as models
    return torchvision.models.inception_v3(pretrained = True, **kwargs)
