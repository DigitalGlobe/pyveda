def ResNet18(**kwargs):
    import torchvision.models as models
    return models.resnet18(pretrained = True, **kwargs)
