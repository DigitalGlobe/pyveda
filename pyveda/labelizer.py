import ipywidgets as widgets
from ipywidgets import Button, HBox, VBox
from IPython.display import display, clear_output
from shapely.geometry.geo import shape
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gbdxtools import Interface 
from shapely.geometry.geo import shape
import numpy as np
import requests

from pyveda import DataPoint

gbdx=Interface()
headers = {"Authorization": "Bearer {}".format(gbdx.gbdx_connection.access_token)}
conn = requests.Session()
conn.headers.update(headers)


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
        self.ids = ids
        self.count = count
        self.imshape = imshape
        self.dtype = dtype
        self.mltype = mltype
        self.index = 0

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
            #next(self.ids)
            self.index += 1
        elif b.description == 'No':
            #figure out how to delete
            #next(self.ids)
            self.index += 1
        elif b.description == 'Exit':
            self.index = self.count
        # TODO should call next i think...
        self.clean()

    def _plot_polygons(self):
        #figure out how to get labels from link
        for pxb in shp:
            if np.size(s)==4:
                ax.add_patch(patches.Rectangle((pxb[0],pxb[1]),(pxb[2]-pxb[0]),\
                        (pxb[3]-pxb[1]),edgecolor='red',fill=False, lw=2))

    def _display_image(self, dp):
        plt.figure(figsize = (7, 7))
        ax = plt.subplot()
        ax.axis("off")
        img = np.rollaxis(dp.image.compute(),0,3)
        ax.imshow(img)
        #self._plot_polygons()
        plt.title('Is this tile correct?')

    def get_next(self):
        try:
            dp_url, img_url = self.ids.__next__()
            r = conn.get(dp_url).json()
            return DataPoint(r, shape=self.imshape, dtype=self.dtype, mltype=self.mltype)
        except Exception as err:
            return None

    def clean(self):
        clear_output()
        buttons = self._create_buttons()
        for b in buttons:
            b.on_click(self._handle_buttons)

        dp = self.get_next()

        if dp is not None:
            print("%0.f tiles out of %0.f tiles have been cleaned" %
                 (self.index, self.count))
            self._display_image(dp)
            display(HBox(buttons))
        else:
            print('all tiles have been cleaned')
