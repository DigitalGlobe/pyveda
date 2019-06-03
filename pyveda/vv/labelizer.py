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
from pyveda.vedaset import veda, abstract
from pyveda import main

class Labelizer():
    def __init__(self, vset, mltype=None, count=None, classes=None, include_background_tiles=None):
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
        self.vedaset = iter(self.vedaset)
        if count is not None:
            self.count = count
        else:
            try:
                self.count = self.vedaset.count
            except:
                self.count = len(self.vedaset)
        self.index = None
        self.mltype = mltype
        self.classes = classes
        self.flagged_tiles = []
        self.iflagged_tiles = []
        self.include_background_tiles = include_background_tiles
        self.id = []
        if isinstance(self.vedaset, store.vedabase.VedaBase):
            self.vb_vcp_id = self.vedaset.dataset_id
            self.vb_vcp = main.from_id(self.vb_vcp_id)
        self._get_next()  #create images, labels, and datapoint


    def _get_next(self):
        if self.index is not None:
            self.index +=1
        else:
            self.index = 0
        self.datapoint = next(self.vedaset)
        if self.include_background_tiles:
            self.image = self._create_images()
            self.labels = self._create_labels()
        else:
            _check_for_background_tile = self._check_for_background_tile()
            if _check_for_background_tile:
                self.image = self._create_images()
                self.labels = self._create_labels()
            else:
                self._get_next()

    def _check_for_background_tile(self):
        lbl = self._create_labels()
        if self.mltype == 'object_detection' or self.mltype == 'segmentation':
            for i, shp in enumerate(lbl):
                if len(shp) is not 0:
                    return True
            else:
                return False
        if self.mltype == 'classification':
            for i, shp in enumerate(lbl):
                if shp != 0:
                    return True
            else:
                return False

    def _create_images(self):
        """
        Creates image tiles from a datapoint
        returns:
            img: An image tile of size (M,N,3)
        """
        if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
            img = self.datapoint.image
        elif isinstance(self.vedaset, (stream.vedastream.BufferedSampleArray,
                      store.vedabase.VedaBase)):
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
        elif isinstance(self.vedaset, (stream.vedastream.BufferedSampleArray,
                      store.vedabase.VedaBase)):
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

    def _create_preview_buttons(self):
        buttons = []
        actions = ['Show next tile set', 'Exit']
        for b in actions:
            btn = Button(description=b)
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
            self._get_next()
        elif b.description == 'No':
            self.flagged_tiles.append(self.datapoint)
            self._get_next()
        elif b.description == 'Exit':
            self.index = self.count
        self.clean()

    def _handle_preview_buttons(self, b):
        if b.description == ('Show next tile set'):
            self.preview()
        elif b.description == 'Exit':
            clear_output()
            return

    def _handle_flag_buttons(self, b):
        """
        Callback and handling of widget buttons for flagged tiles.
        """
        try:
            if b.description == 'Keep':
                self.datapoint = next(self.iflagged_tiles)
                self.image = self._create_images()
                self.labels = self._create_labels()
            elif b.description == 'Remove':
                self.remove_dp()
                self.datapoint = next(self.iflagged_tiles)
                self.image = self._create_images()
                self.labels = self._create_labels()
            self.clean_flags()
        except StopIteration:
            print("All flagged tiles have been cleaned.")

    def remove_dp(self):
        if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
            self.datapoint.remove()
        elif isinstance(self.vedaset,  store.vedabase.VedaBase):
            vb_dp_id = self.vedaset.metadata[self.index][0].decode('utf-8')
            print(vb_dp_id)
            vb_dp = self.vb_vcp.fetch_sample_from_id(vb_dp_id)
            vb_dp.remove()

    def _recolor_images(self):
        img = self.image.astype('float32')
        img[:,:,0] /= np.max(img[:,:,0])
        img[:,:,1] /= np.max(img[:,:,1])
        img[:,:,2] /= np.max(img[:,:,2])
        return(img)

    def _display_image(self):
        """
        Displays image tile for a given vedaset object.
        """
        if self.image.dtype == 'uint16':
            img = self._recolor_images()
        else:
            img = self.image
        plt.figure(figsize = (10, 10))
        self.ax = plt.subplot()
        self.ax.axis("off")
        self.ax.imshow(img)

    def _display_obj_detection(self, title=True):
        """
        Adds vedaset object detection label geometries to the image tile plot.
        """
        if title==True:
            plt.title('Is this tile correct?', fontsize=14)
        legend_elements = []
        for i, shp in enumerate(self.labels):
            if len(shp) is not 0:
                edge_color = np.random.rand(3,)
                handle = patches.Patch(edgecolor=edge_color, fill=False, label = self.classes[i])
                legend_elements.append(handle)
                self.ax.legend(handles=legend_elements, loc='lower center',
                         bbox_to_anchor=(0.5,-0.1), ncol=3, fancybox=True, fontsize=12)
                for pxb in shp:
                    self.ax.add_patch(patches.Rectangle((pxb[0],pxb[1]),(pxb[2]-pxb[0]),\
                            (pxb[3]-pxb[1]),edgecolor=edge_color,
                            fill=False, lw=2))


    def _display_classification(self, title=True):
        """
        Adds vedaset classification labels to the image plot.
        """
        positive_classes = []
        for i, binary_class in enumerate(self.labels):
            if binary_class != 0:
                positive_classes.append(self.classes[i])
        if title==True:
            if positive_classes:
                plt.title('Does this tile contain: %s?' % ', '.join(positive_classes), fontsize=14)
            else:
                plt.title('Does this tile contain: no labeled objects?', fontsize=14)
        else:
            if positive_classes:
                plt.title('Tile contains: %s' % ', '.join(positive_classes), fontsize=14)
            else:
                plt.title('Tile contains: no labeled objects', fontsize=14)

    def _display_segmentation(self, title=True):
        """
        Adds vedaset classification labels to the image plot.
        """
        if title==True:
            plt.title('Is this tile correct?', fontsize=14)
        if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
            legend_elements = []
            for i, shp in enumerate(self.labels):
                if len(shp) is not 0:
                    face_color = np.random.rand(3,)
                    handle = patches.Patch(color=face_color, label = self.classes[i])
                    legend_elements.append(handle)
                    self.ax.legend(handles=legend_elements, loc='lower center',
                             bbox_to_anchor=(0.5,-0.1), ncol=3, fancybox=True, fontsize=12)
                for coord in shp:
                    if coord['type']=='Polygon':
                        geom = Polygon(coord['coordinates'][0])
                        x,y = geom.exterior.xy
                        self.ax.fill(x,y, color=face_color, alpha=0.4)
                        self.ax.plot(x,y, lw=3, color=face_color)
        else:
            legend_colors = [(0.5,0.5,0.5,0)]
            cmap_name = 'segmentation_labels'
            for i, shp in enumerate(self.classes):
                color = np.random.rand(3,)
                legend_colors.append(color)
            cm = LinearSegmentedColormap.from_list(cmap_name, legend_colors, N=100)
            im = self.ax.imshow(self.labels, alpha=0.5, cmap=cm)
            values = np.unique(self.labels.ravel())
            colors = [im.cmap(im.norm(value)) for value in values]
            try:
                lpatches = [patches.Patch(color=colors[i+1], label=a) for i,a in enumerate(self.classes)]
                self.ax.legend(handles=lpatches, bbox_to_anchor=(0.5,-0.1))
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
            if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
                if self.mltype == 'object_detection':
                    self._display_obj_detection()
                if self.mltype == 'classification':
                    self._display_classification()
                if self.mltype == 'segmentation':
                    self._display_segmentation()
            else:
                if isinstance(self.mltype, abstract.BinaryClassificationType):
                    self._display_classification()
                if isinstance(self.mltype, abstract.ObjectDetectionType):
                    self._display_obj_detection()
                if isinstance(self.mltype, abstract.InstanceSegmentationType):
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
            if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
                if self.mltype == 'object_detection':
                    self._display_obj_detection()
                if self.mltype == 'classification':
                    self._display_classification()
                if self.mltype == 'segmentation':
                    self._display_segmentation()
            else:
                if isinstance(self.mltype, abstract.BinaryClassificationType):
                    self._display_classification()
                if isinstance(self.mltype, abstract.ObjectDetectionType):
                    self._display_obj_detection()
                if isinstance(self.mltype, abstract.InstanceSegmentationType):
                    self._display_segmentation()
        else:
            try:
                print("You've flagged %0.f bad tiles. Review them now" %len(self.flagged_tiles))
                self.iflagged_tiles = iter(self.flagged_tiles)
                self.datapoint = next(self.iflagged_tiles)
                self.image = self._create_images()
                self.labels = self._create_labels()
                self.clean_flags()
            except StopIteration:
                print("All tiles have been cleaned.")

    def preview(self):
        clear_output()
        buttons = self._create_preview_buttons()
        for b in buttons:
            b.on_click(self._handle_preview_buttons)
        display(HBox(buttons))
        for c in range(0, self.count):
            self._display_image()
            if isinstance(self.vedaset, veda.api.VedaCollectionProxy):
                if self.mltype == 'object_detection':
                    self._display_obj_detection(title=False)
                if self.mltype == 'classification':
                    self._display_classification(title=False)
                if self.mltype == 'segmentation':
                    self._display_segmentation(title=False)
            else:
                if isinstance(self.mltype, abstract.BinaryClassificationType):
                    self._display_classification(title=False)
                if isinstance(self.mltype, abstract.ObjectDetectionType):
                    self._display_obj_detection(title=False)
                if isinstance(self.mltype, abstract.InstanceSegmentationType):
                    self._display_segmentation(title=False)
            plt.show()
            self._get_next()

    def remove_black_tiles(self):
        for c in range(self.count):
            if np.amax(self.image) == 0:
                self.datapoint.remove()
                print('removing tile %s' % self.datapoint.id)
            self._get_next()
