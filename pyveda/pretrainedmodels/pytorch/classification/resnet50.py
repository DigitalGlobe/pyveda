def ResNet50(**kwargs):
    import torchvision.models as models
    return models.resnet50(pretrained=True, **kwargs)
