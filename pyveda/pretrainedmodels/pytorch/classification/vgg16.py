def VGG16(**kwargs):
    import torchvision.models as models
    return models.vgg16(pretrained=True, **kwargs)
