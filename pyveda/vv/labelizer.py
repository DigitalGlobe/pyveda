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

from pyveda.auth import Auth

gbdx = Auth()
conn = gbdx.gbdx_connection


class Labelizer():
    def __init__(self, ids, count, imshape, dtype, mltype):
        """
          Labelizer will page through image/labels and allow users to remove/change data or labels from a VedaCollection
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

        self.ids = ids
        self.count = count
        self.imshape = imshape
        self.dtype = dtype
        self.mltype = mltype
        self.index = 0
        self.dp = self.get_next()
        self.flagged_tiles = []

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

    def _handle_buttons(self, b):
        """
        Callback and handling of widget buttons.
        """
        if b.description == 'Yes':
            self.index += 1
            self.dp = self.get_next()
        elif b.description == 'No':
            self.flagged_tiles.append(self.dp)
            self.index += 1
            self.dp = self.get_next()
        elif b.description == 'Exit':
            self.index = self.count
        self.clean()

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

    def _handle_flag_buttons(self, b):
        """
        Callback and handling of widget buttons for flagged tiles.
        """
        try:
            if b.description == 'Keep':
                print('dp %s has been stored' %self.dp.id)
                self.dp = next(self.flagged_tiles)
            elif b.description == 'Remove':
                self.dp.remove()
                print('dp %s has been removed' %self.dp.id)
                self.dp = next(self.flagged_tiles)
            self.clean_flags()
        except StopIteration:
            print("All flagged tiles have been cleaned.")

    def _display_obj_detection(self, dp):
        """
        Adds DataPoint object detection label geometries to the image tile plot.
        Params:
        dp: A DataPoint object for the VedaCollection.
        """
        label = list(dp.label.items())
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



    def _display_classification(self, dp):
        """
        Adds DataPoint classification labels to the image plot.
        Params:
        dp: A DataPoint object for the VedaCollection.
        """
        label = list(dp.label.items())
        label_class = [l[1] for l in label]
        label_type = [l[0] for l in label]
        positive_classes = []
        for i, binary_class in enumerate(label_class):
            if binary_class != 0:
                positive_classes.append(label_type[i])
        plt.title('Does this tile contain: %s?' % ', '.join(positive_classes), fontsize=14)

    def _display_segmentation(self, dp):
        """
        Adds DataPoint classification labels to the image plot.
        Params:
        dp: A DataPoint object for the VedaCollection.
        """
        label = list(dp.label.items())
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

    def _display_image(self, dp):
        """
        Displays image tile for a given DataPoint object.
        Params:
           dp: A DataPoint object for the VedaCollection.
        """
        plt.figure(figsize = (7, 7))
        ax = plt.subplot()
        ax.axis("off")
        img = dp.image.compute()
        try:
            img /= img.max()
        except TypeError:
            img = img
        ax.imshow(np.moveaxis(img, 0, -1))

    def get_next(self):
        """
        Fetches the next DataPoint object from VedaCollection ids.
        """
        try:
            dp_url, img_url = self.ids.__next__()
            r = conn.get(dp_url).json()
            self.dp = DataPoint(r, shape=self.imshape, dtype=self.dtype, mltype=self.mltype)
            return self.dp
        except Exception as err:
            return None

    def clean_flags(self):
        """
        Method for verifying DataPoints that were flagged with clean()
        """
        buttons = self._create_flag_buttons()
        for b in buttons:
            b.on_click(self._handle_flag_buttons)
        if self.dp is not None:
            self._display_image(self.dp)
            if self.dp.mltype == 'object_detection':
                self._display_obj_detection(self.dp)
            if self.dp.mltype == 'classification':
                self._display_classification(self.dp)
            if self.dp.mltype == 'segmentation':
                self._display_segmentation(self.dp)
            print('Do you want to remove this tile?')
            display(HBox(buttons))

    def clean(self):
        """
        Method for verifying each DataPoint as image data with associated polygons.
        Displays a polygon overlayed on image chip with associated ipywidget
        buttons. Allows user to click through each DataPoint object and decide
        whether to keep or remove the object.
        """

        clear_output()
        buttons = self._create_buttons()
        for b in buttons:
            b.on_click(self._handle_buttons)
        if self.dp is not None and self.index != self.count:
            print("%0.f tiles out of %0.f tiles have been cleaned" %
                 (self.index, self.count))
            display(HBox(buttons))
            self._display_image(self.dp)
            if self.dp.mltype == 'object_detection':
                self._display_obj_detection(self.dp)
            if self.dp.mltype == 'classification':
                self._display_classification(self.dp)
            if self.dp.mltype == 'segmentation':
                self._display_segmentation(self.dp)
        else:
            try:
                print("You've flagged %0.f bad tiles. Review them now" %len(self.flagged_tiles))
                self.flagged_tiles = iter(self.flagged_tiles)
                self.dp = next(self.flagged_tiles)
                self.clean_flags()
            except StopIteration:
                print("All tiles have been cleaned.")
