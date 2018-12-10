def DenseNet161(**kwargs):
    import torchvision.models as models
    return models.densenet161(pretrained = True, **kwargs)
