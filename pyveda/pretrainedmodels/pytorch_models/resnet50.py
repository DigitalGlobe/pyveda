def ResNet50(**kwargs):
    import torchvision.models as models
    return torchvision.models.resnet50(pretrained = True, **kwargs)
