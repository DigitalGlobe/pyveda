def AlexNet(**kwargs):
    import torchvision.models as models
    return torchvision.models.alexnet(pretrained = True, **kwargs)
