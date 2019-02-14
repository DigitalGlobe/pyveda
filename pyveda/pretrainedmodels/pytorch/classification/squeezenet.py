def SqueezeNet(**kwargs):
    import torchvision.models as models
    return torchvision.models.squeezenet(pretrained=True, **kwargs)
