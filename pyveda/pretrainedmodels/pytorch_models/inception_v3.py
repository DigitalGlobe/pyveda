def InceptionV3(**kwargs):
    import torchvision.models as models
    return models.inception_v3(pretrained = True, **kwargs)
