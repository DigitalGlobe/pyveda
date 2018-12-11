try:
    import torch
except ImportError:
    raise ImportError('pytorch not installed. Please install pytorch to import models')

try:
    import torchvision
except ImportError:
    raise ImportError('torvision not installed. Please install torchvision to import models.')

from pyveda.pretrainedmodels.pytorch.resnet50 import ResNet50
from pyveda.pretrainedmodels.pytorch.vgg16 import VGG16
from pyveda.pretrainedmodels.pytorch.inception_v3 import InceptionV3
from pyveda.pretrainedmodels.pytorch.densenet161 import DenseNet161
from pyveda.pretrainedmodels.pytorch.alexnet import AlexNet
from pyveda.pretrainedmodels.pytorch.squeezenet import SqueezeNet
from pyveda.pretrainedmodels.pytorch.resnet18 import ResNet18

from pyveda.pretrainedmodels.pytorch.ternausnet_unet import unet11
from pyveda.pretrainedmodels.pytorch.ternausnet_v2 import TernausNetV2
