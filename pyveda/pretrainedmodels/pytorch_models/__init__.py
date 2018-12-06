try:
    import pytorch
except ImportError:
    raise ImportError('pytorch not installed. Please install pytorch to import models')

from pyveda.pretrainedmodels.pytorch_models.resnet50 import ResNet50
