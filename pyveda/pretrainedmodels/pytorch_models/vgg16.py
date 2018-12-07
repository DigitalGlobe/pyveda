def VGG16(**kwargs):
    import torchvision.models as models
    return torchvision.models.vgg16(pretrained = True, **kwargs)
