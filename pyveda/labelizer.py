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
        self.flagged_tiles=[]

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
            self.flagged_tiles.append(self.ids)
            self.index += 1
            self.dp = self.get_next()
        elif b.description == 'Exit':
            self.index = self.count
        self.clean()

    def _create_flag_buttons(self):
        """
        Creates ipywidget widget buttons for flagged tiles
        Returns:
            radio_buttons: A list of ipywidget button() objects
        """
        buttons = []
        actions = [('Keep', 'success'), ('Remove', 'danger')]
        for b in actions:
            btn = Button(description=b[0], button_style=b[1])
            buttons.append(btn)
        return buttons

    def _handle_flag_buttons(self, b):
        id = iter(self.flagged_tiles)
        if b.description == 'Keep':
            print('dp %s has been stored' %self.dp.id)
            self.dp = self.get_next()
        elif b.description == 'Remove':
            self.dp.remove()
            print('dp %s has been removed' %self.dp.id)
            self.dp = self.get_next()
        self.clean_flags()


    def _display_polygons(self):
        label = list(self.dp.label.items())
        label_shp = [l[1] for l in label]
        label_type = [l[0] for l in label]
        ax = plt.subplot()
        for i,shp in enumerate(label_shp):
            if len(shp) is not 0:
                face_color = np.random.rand(3,)
                for pxb in shp:
                    ax.add_patch(patches.Rectangle((pxb[0],pxb[1]),(pxb[2]-pxb[0]),\
                            (pxb[3]-pxb[1]),edgecolor=face_color,
                            fill=False, lw=2, label=label_type[i]))
                #ax.legend() ##TODO: figure out optimal legend/label formatting.


    def _display_image(self):
        plt.figure(figsize = (7, 7))
        ax = plt.subplot()
        ax.axis("off")
        img = np.rollaxis(self.dp.image.compute(),0,3)
        ax.imshow(img)

    def get_next(self):
        # try:
        dp_url, img_url = self.ids.__next__()
        r = conn.get(dp_url).json()
        return DataPoint(r, shape=self.imshape, dtype=self.dtype, mltype=self.mltype)
        # except Exception as err:
        #     #print('no more DataPoints')
        #     print(err)
        #     return None

    def clean_flags(self):
            buttons = self._create_flag_buttons()
            for b in buttons:
                b.on_click(self._handle_flag_buttons)
            if self.dp is not None:
                self._display_image()
                self._display_polygons()
                display(HBox(buttons))


    def clean(self):
        clear_output()
        buttons = self._create_buttons()
        for b in buttons:
            b.on_click(self._handle_buttons)
        if self.dp is not None and self.index != self.count:
            print("%0.f tiles out of %0.f tiles have been cleaned" %
                 (self.index, self.count))
            self._display_image()
            self._display_polygons()
            plt.title('Is this tile correct?')
            display(HBox(buttons))
        else:
            ##todo: add conditionals and index to exit when all tiles have been flagged
            print("You've flagged %0.f bad tiles. Review them now" %len(self.flagged_tiles))
            self.ids = iter(self.flagged_tiles)
            # self.dp = self.get_next()
            self.clean_flags()
