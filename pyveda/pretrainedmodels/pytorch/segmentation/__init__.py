try:
    import torch
except ImportError:
    raise ImportError('pytorch not installed. Please install pytorch to import models')

try:
    import torchvision
except ImportError:
    raise ImportError('torvision not installed. Please install torchvision to import models.')


from pyveda.pretrainedmodels.pytorch.segmentation.ternausnet_unet import unet11

#from pyveda.pretrainedmodels.pytorch.segmentation.ternausnet_v2 import TernausNetV2
