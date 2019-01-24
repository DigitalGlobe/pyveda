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
# from pyveda.vcp.stream import vedastream
from pyveda.auth import Auth
from pyveda.veda import api

gbdx = Auth()
conn = gbdx.gbdx_connection

class Labelizer():
    def __init__(self, vcp, mltype, ids):
        """
          Labelizer will page through image/labels and allow users to remove/change data or labels from a VedaBase or VedaStream
          Params:
            vcp: The data to be cleaned
        """
        assert has_ipywidgets, 'Labelizer requires ipywidgets to be installed'
        assert has_ipy, 'Labelizer requires ipython to be installed'
        assert has_plt, 'Labelizer requires matplotlib to be installed'

        self.vcp = vcp
        self.ids = ids
        self.count = self.vcp.count
        self.index = 0
        self.mltype = mltype
        self.datapoint = self.vcp[self.index]
        self.image = self.datapoint.image
        self.labels = self.datapoint.label
        self.flagged_tiles = []
        # self.classes = classes
        # self.labels = self.datapoint[0]

    def _create_buttons(self):
        """
        Creates ipywidget widget buttons
        Returns:
            buttons: A list of ipywidget Button() objects
        """
        buttons = []
        actions = [('Yes', 'success'), ('No', 'danger'), ('Exit', 'info' )]
        for b in actions:
            btn = Button(description=b[0], button_style=b[1])
            buttons.append(btn)
        return buttons

    def _create_flag_buttons(self):
        """
        Creates ipywidget widget buttons for tiles that have been flagged for review.
        Returns:
            buttons: A list of ipywidget Button() objects
        """
        buttons = []
        actions = [('Keep', 'success'), ('Remove', 'danger')]
        for b in actions:
            btn = Button(description=b[0], button_style=b[1])
            buttons.append(btn)
        return buttons

    def _handle_buttons(self, b):
        """
        Callback and handling of widget buttons.
        """
        if b.description == 'Yes':
            self.index += 1
            self.datapoint = self.vcp[self.index]
            self.image = self.datapoint.image
            self.labels = self.datapoint.label
        elif b.description == 'No':
            self.flagged_tiles.append(self.datapoint)
            self.index += 1
            self.datapoint = self.vcp[self.index]
            self.image = self.datapoint.image
            self.labels = self.datapoint.label
        elif b.description == 'Exit':
            self.index = self.count
        self.clean()

    def _handle_flag_buttons(self, b):
        """
        Callback and handling of widget buttons for flagged tiles.
        """
        try:
            if b.description == 'Keep':
                self.datapoint = next(self.flagged_tiles)
                self.image = self.datapoint.image
                self.labels = self.datapoint.label
            elif b.description == 'Remove':
                #TODO: add actual removal of point!
                self.datapoint = next(self.flagged_tiles)
                self.image = self.datapoint.image
                self.labels = self.datapoint.label
            self.clean_flags()
        except StopIteration:
            print("All flagged tiles have been cleaned.")

    def _display_image(self):
        """
        Displays image tile for a given vcp object.
        """
        img = self.image
        try:
            img = img/np.amax(img)
        except TypeError:
            img = img
        plt.figure(figsize = (10, 10))
        ax = plt.subplot()
        ax.axis("off")
        ax.imshow(img)
        return(img)

    def _display_obj_detection(self):
        """
        Adds vcp object detection label geometries to the image tile plot.
        """
        label = list(self.labels.items())
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


    def _display_classification(self):
        """
        Adds vcp classification labels to the image plot.
        """
        label = list(self.labels.items())
        label_shp = [l[1] for l in label]
        label_type = [l[0] for l in label]
        positive_classes = []
        for i, binary_class in enumerate(label_shp):
            if binary_class != 0:
                positive_classes.append(label_type[i])
        plt.title('Does this tile contain: %s?' % ', '.join(positive_classes), fontsize=14)

    def _display_segmentation(self):
        """
        Adds vcp classification labels to the image plot.
        """
        label = list(self.labels.items())
        label_shp = [l[1] for l in label]
        label_type = [l[0] for l in label]
        legend_elements = []
        ax = plt.subplot()
        plt.title('Is this tile correct?', fontsize=14)
        for i, shp in enumerate(label_shp):
            if len(shp) is not 0:
                face_color = np.random.rand(3,)
                handle = patches.Patch(color=face_color, label = label_type[i])
                legend_elements.append(handle)
                ax.legend(handles=legend_elements, loc='lower center',
                         bbox_to_anchor=(0.5,-0.1), ncol=3, fancybox=True, fontsize=12)
            for coord in shp:
                if coord['type']=='Polygon':
                    geom = Polygon(coord['coordinates'][0])
                    x,y = geom.exterior.xy
                    ax.fill(x,y, color=face_color, alpha=0.4)
                    ax.plot(x,y, lw=3, color=face_color)

    def clean_flags(self):
        """
        Method for verifying DataPoints that were flagged with clean()
        """
        clear_output()
        buttons = self._create_flag_buttons()
        for b in buttons:
            b.on_click(self._handle_flag_buttons)
        if self.datapoint is not None:
            self._display_image()
            if self.mltype == 'object_detection':
                self._display_obj_detection()
            if self.mltype == 'classification':
                self._display_classification()
            if self.mltype == 'segmentation':
                self._display_segmentation()
            print('Do you want to remove this tile?')
            display(HBox(buttons))

    def clean(self):
        """
        Method for verifying each vcp object as image data with associated polygons.
        Displays a polygon overlayed on image chip with associated ipywidget
        buttons. Allows user to click through each vcp object and decide
        whether to keep or remove the object.
        """
        clear_output()
        buttons = self._create_buttons()
        for b in buttons:
            b.on_click(self._handle_buttons)
        if self.image is not None and self.index != self.count:
            print("%0.f tiles out of %0.f tiles have been cleaned" %
                 (self.index, self.count))
            display(HBox(buttons))
            self._display_image()
            if self.mltype == 'object_detection':
                self._display_obj_detection()
            if self.mltype == 'classification':
                self._display_classification()
            if self.mltype == 'segmentation':
                self._display_segmentation()
        else:
            try:
                print("You've flagged %0.f bad tiles. Review them now" %len(self.flagged_tiles))
                self.flagged_tiles = iter(self.flagged_tiles)
                self.datapoint = next(self.flagged_tiles)
                self.image = self.datapoint.image
                self.labels = self.datapoint.label
                self.clean_flags()
            except StopIteration:
                print("All tiles have been cleaned.")
