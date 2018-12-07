def DenseNet161(**kwargs):
    import torchvision.models as models
    return torchvision.models.densenet161(pretrained = True, **kwargs)
