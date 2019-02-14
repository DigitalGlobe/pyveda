try:
    import torch
except ImportError:
    raise ImportError(
        'pytorch not installed. Please install pytorch to import models')

try:
    import torchvision
except ImportError:
    raise ImportError(
        'torvision not installed. Please install torchvision to import models.')

from pyveda.pretrainedmodels.pytorch.classification.resnet50 import ResNet50
from pyveda.pretrainedmodels.pytorch.classification.vgg16 import VGG16
from pyveda.pretrainedmodels.pytorch.classification.inception_v3 import InceptionV3
from pyveda.pretrainedmodels.pytorch.classification.densenet161 import DenseNet161
from pyveda.pretrainedmodels.pytorch.classification.alexnet import AlexNet
from pyveda.pretrainedmodels.pytorch.classification.squeezenet import SqueezeNet
from pyveda.pretrainedmodels.pytorch.classification.resnet18 import ResNet18
