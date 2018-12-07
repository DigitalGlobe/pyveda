try:
    import torch
except ImportError:
    raise ImportError('pytorch not installed. Please install pytorch to import models')

from pyveda.pretrainedmodels.pytorch_models.resnet50 import ResNet50
from pyveda.pretrainedmodels.pytorch_models.vgg16 import VGG16
from pyveda.pretrainedmodels.pytorch_models.inception_v3 import InceptionV3
from pyveda.pretrainedmodels.pytorch_models.densenet161 import DenseNet161
from pyveda.pretrainedmodels.pytorch_models.alexnet import AlexNet
from pyveda.pretrainedmodels.pytorch_models.squeezenet import SqueezeNet
