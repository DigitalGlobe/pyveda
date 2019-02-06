try:
    import ipywidgets as widgets
    from ipywidgets import Button, HBox, VBox, interact, IntSlider
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
    from matplotlib.colors import LinearSegmentedColormap
    has_plt = True
except:
    has_plt = False

from shapely.geometry.geo import shape
from shapely.geometry import *
import numpy as np
import requests
from pyveda.auth import Auth
from pyveda.vedaset import stream, store
from pyveda import veda

class Labelizer():
    def __init__(self, vset, mltype, count, classes):
        """
          Labelizer will page through image/labels and allow users to remove/change data or labels from a VedaBase or VedaStream
          Params:
            vset: The data to be cleaned
            mltype: the type of ML data. Can be 'classification' 'segmentation' or 'object_detection'
            count: the number of datapoints to iterate through
            classes: the classes in the dataset
        """
        assert has_ipywidgets, 'Labelizer requires ipywidgets to be installed'
        assert has_ipy, 'Labelizer requires ipython to be installed'
        assert has_plt, 'Labelizer requires matplotlib to be installed'

        self.vedaset = vset
        if count is not None:
            self.count = count
        else:
            try:
                self.count = self.vedaset.count
            except:
                self.count = len(self.vedaset)
        self.index = 0
        self.mltype = mltype
        self.datapoint = self.vedaset[self.index]
        self.image = self._create_images()
        self.classes = classes
        self.labels = self._create_labels()
        self.flagged_tiles = []

    def _create_images(self):
        """
        Creates image tiles from a datapoint
        returns:
            img: An image tile of size (M,N,3)
        """
        if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
            img = self.datapoint.image
        if isinstance(self.vedaset, (stream.vedastream.BufferedSampleArray,
                      store.vedabase.WrappedDataNode)):
            img = np.moveaxis(self.datapoint[0], 0, -1)
        return img

    def _create_labels(self):
        """
        Generates labels for a datapoint's image tile
        returns:
            labels: a list of labels for an image tile
        """
        if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
            labels = self.datapoint.label.values()
        if isinstance(self.vedaset, (stream.vedastream.BufferedSampleArray,
                      store.vedabase.WrappedDataNode)):
            labels = self.datapoint[1]
        return labels

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
            self.datapoint = self.vedaset[self.index]
            self.image = self._create_images()
            self.labels = self._create_labels()
        elif b.description == 'No':
            self.flagged_tiles.append(self.datapoint)
            self.index += 1
            self.datapoint = self.vedaset[self.index]
            self.image = self._create_images()
            self.labels = self._create_labels()
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
                self.image = self._create_images()
                self.labels = self._create_labels()
            elif b.description == 'Remove':
                self.datapoint.remove() ##only works for VCP, currently
                self.datapoint = next(self.flagged_tiles)
                self.image = self._create_images()
                self.labels = self._create_labels()
            self.clean_flags()
        except StopIteration:
            print("All flagged tiles have been cleaned.")

    def _display_image(self):
        """
        Displays image tile for a given vedaset object.
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
        # return(img)

    def _display_obj_detection(self):
        """
        Adds vedaset object detection label geometries to the image tile plot.
        """
        legend_elements = []
        ax = plt.subplot()
        plt.title('Is this tile correct?', fontsize=14)
        for i, shp in enumerate(self.labels):
            if len(shp) is not 0:
                edge_color = np.random.rand(3,)
                handle = patches.Patch(edgecolor=edge_color, fill=False, label = self.classes[i])
                legend_elements.append(handle)
                ax.legend(handles=legend_elements, loc='lower center',
                         bbox_to_anchor=(0.5,-0.1), ncol=3, fancybox=True, fontsize=12)
                for pxb in shp:
                    ax.add_patch(patches.Rectangle((pxb[0],pxb[1]),(pxb[2]-pxb[0]),\
                            (pxb[3]-pxb[1]),edgecolor=edge_color,
                            fill=False, lw=2))


    def _display_classification(self):
        """
        Adds vedaset classification labels to the image plot.
        """
        positive_classes = []
        for i, binary_class in enumerate(self.labels):
            if binary_class != 0:
                positive_classes.append(self.classes[i])
        plt.title('Does this tile contain: %s?' % ', '.join(positive_classes), fontsize=14)

    def _display_segmentation(self):
        """
        Adds vedaset classification labels to the image plot.
        """
        ax = plt.subplot()
        plt.title('Is this tile correct?', fontsize=14)
        if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
            legend_elements = []
            for i, shp in enumerate(self.labels):
                if len(shp) is not 0:
                    face_color = np.random.rand(3,)
                    handle = patches.Patch(color=face_color, label = self.classes[i])
                    legend_elements.append(handle)
                    ax.legend(handles=legend_elements, loc='lower center',
                             bbox_to_anchor=(0.5,-0.1), ncol=3, fancybox=True, fontsize=12)
                for coord in shp:
                    if coord['type']=='Polygon':
                        geom = Polygon(coord['coordinates'][0])
                        x,y = geom.exterior.xy
                        ax.fill(x,y, color=face_color, alpha=0.4)
                        ax.plot(x,y, lw=3, color=face_color)
        else:
            legend_colors = [(0.5,0.5,0.5)]
            cmap_name = 'segmentation_labels'
            for i, shp in enumerate(self.classes):
                color = np.random.rand(3,)
                legend_colors.append(color)
            cm = LinearSegmentedColormap.from_list(cmap_name, legend_colors, N=100)
            im = ax.imshow(self.labels, alpha=0.5, cmap=cm)
            values = np.unique(self.labels.ravel())
            colors = [im.cmap(im.norm(value)) for value in values]
            try:
                lpatches = [patches.Patch(color=colors[i+1], label=a) for i,a in enumerate(self.classes)]
                ax.legend(handles=lpatches, bbox_to_anchor=(0.5,-0.1))
            except:
                pass

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
        Method for verifying each vedaset object as image data with associated polygons.
        Displays a polygon overlayed on image chip with associated ipywidget
        buttons. Allows user to click through each vedaset object and decide
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
                self.image = self._create_images()
                self.labels = self._create_labels()
                self.clean_flags()
            except StopIteration:
                print("All tiles have been cleaned.")


    def preview(self):
        for c in range(self.count):
            self._display_image()
            if self.mltype == 'object_detection':
                self._display_obj_detection()
            if self.mltype == 'classification':
                self._display_classification()
            if self.mltype == 'segmentation':
                self._display_segmentation()
            plt.show()
            self.index += 1
            self.datapoint = self.vedaset[self.index]
            self.image = self._create_images()
            self.labels = self._create_labels()
