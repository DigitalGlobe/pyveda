try:
    import ipywidgets as widgets
    from ipywidgets import Button, HBox, VBox, RadioButtons
    has_ipywidgets = True
except:
    has_ipywidgets = False

try:
    from IPython.display import display, clear_output
    has_ipy = True
except:
    has_ipy = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    has_plt = True
except:
    has_plt = False

from shapely.geometry.geo import shape
from shapely.geometry import *
import numpy as np
import requests
from pyveda.vedaset.stream import vedastream
from pyveda.auth import Auth
from pyveda.veda import api

gbdx = Auth()
conn = gbdx.gbdx_connection

class Labelizer():
    def __init__(self, vedaset, count):
        """
          Labelizer will page through image/labels and allow users to remove/change data or labels from a VedaBase or VedaStream
          Params:
            ids (generator): A Url generator for datapoints to be viewed
            count (int): the number of datapoints to process
            imshape (tuple): shape of each incoming image
            dtype (str): the datatype of the images
            mltype (str): the mltype of the veda collection
        """
        assert has_ipywidgets, 'Labelizer requires ipywidgets to be installed'
        assert has_ipy, 'Labelizer requires ipython to be installed'
        assert has_plt, 'Labelizer requires matplotlib to be installed'

        # self.ids = ids #not sure if we need or not yet.
        self.vedaset = vedaset
        # self.count = count
        # self.imshape = imshape
        self.dtype = next(self.vedaset)[0]['properties']['dtype']
        self.mltype = next(self.vedaset)[0]['properties']['mltype']
        self.index = 0
        self.image = next(self.vedaset)[1]
        self.props = next(self.vedaset)[0]
        # self.flagged_tiles = []


    def _display_image(self):
        """
        Displays image tile for a given DataPoint object.
        """
        img = self.image
        plt.figure(figsize = (7, 7))
        ax = plt.subplot()
        ax.axis("off")
        ax.imshow(np.moveaxis(img, 0, -1))
        return(img)

    def _display_obj_detection(self):
        """
        Adds DataPoint object detection label geometries to the image tile plot.
        Params:
        dp: A DataPoint object for the VedaCollection.
        """
        label = self.props['properties']['label'].items()
        label_shp = [l[1] for l in label]
        label_type = [l[0] for l in label]
        legend_elements = []
        ax = plt.subplot()
        plt.title('Is this tile correct?', fontsize=14)
        for i,shp in enumerate(label_shp):
            if len(shp) is not 0:
                edge_color = np.random.rand(3,)
                handle = patches.Patch(edgecolor=edge_color, fill=False, label = label_type[i])
                legend_elements.append(handle)
                ax.legend(handles=legend_elements, loc='lower center',
                         bbox_to_anchor=(0.5,-0.1), ncol=3, fancybox=True, fontsize=12)
                for pxb in shp:
                    ax.add_patch(patches.Rectangle((pxb[0],pxb[1]),(pxb[2]-pxb[0]),\
                            (pxb[3]-pxb[1]),edgecolor=edge_color,
                            fill=False, lw=2))

    def clean(self):
        self._display_image()
        self._display_obj_detection()
        # return(a)
