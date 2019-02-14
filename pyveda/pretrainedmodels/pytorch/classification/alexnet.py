def AlexNet(**kwargs):
    import torchvision.models as models
    return models.alexnet(pretrained=True, **kwargs)
