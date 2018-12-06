try:
    import pytorch
except ImportError:
    raise ImportError('pytorch not installed. Please install pytorch to import models')

def ResNet50(**kwargs):
    import torchvision.models as models
    return torchvision.models.resnet50(pretrained = True, **kwargs)
