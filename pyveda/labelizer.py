import ipywidgets as widgets
from ipywidgets import Button, HBox, VBox
from IPython.display import display, clear_output
from rasterio import features
from shapely.geometry.geo import shape
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gbdxtools import Interface, TmsImage, CatalogImage, IpeImage
gbdx=Interface()
from rasterio import features
from shapely.geometry.geo import shape
import dask.array as da
from .loaders import load_image


class Labelizer():
    def __init__(self, links, count, conn):
        self.links = links
        self.count = count
        self.conn = conn
        self.index = 0

    def _create_buttons(self):
        """
        Creates ipywidget widget buttons
        Returns:
            buttons: A list of ipywidget Button() objects
        """
        buttons = []
        actions = [('Yes', 'success'), ('No', 'danger')]
        for b in actions:
            btn = Button(description=b[0], button_style=b[1])
            buttons.append(btn)
        return buttons

    def _handle_buttons(self, b):
        """
        Callback and handling of widget buttons.
        """
        if b.description == 'Yes':
            next(self.links)
            index += 1
        elif b.description == 'No':
            self.conn.delete(self.links["delete"]["href"])
            next(self.links)
            index += 1
        self.clean()

    def _plot_polygons(self):
        #figure out how to get labels from link
        for pxb in shp:
            if np.size(s)==4:
                ax.add_patch(patches.Rectangle((pxb[0],pxb[1]),(pxb[2]-pxb[0]),\
                        (pxb[3]-pxb[1]),edgecolor='red',fill=False, lw=2))

    def _compute_image(self):
        token = gbdx.gbdx_connection.access_token
        load = load_image(self.links["image"]["href"], token, self.imshape,
                          dtype=self.dtype)
        dask_array = da.from_delayed(load, shape=self.imshape, dtype=self.dtype)
        return dask_array.compute()

    def _display_image(self):
        plt.figure(figsize = (7, 7))
        ax = plt.subplot()
        ax.axis("off")
        image = _compute_image()
        img = np.rollaxis(image,0,3)
        ax.imshow(img)
        #self._plot_polygons()
        plt.title('Is this tile correct?')

     def _clean(self):
        clear_output()
        buttons = self._create_buttons()
        for b in buttons:
            b.on_click(self._handle_buttons)
        if self.index < self.count:
            print("%0.f tiles out of %0.f tiles have been cleaned" %
                 (self.index, self.count))
            display(HBox(buttons))
            self._display_image()

        if self.index >= len(self.count):
            print('all tiles have been cleaned')
