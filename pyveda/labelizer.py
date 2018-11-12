try:
    import ipywidgets as widgets
    from ipywidgets import Button, HBox, VBox
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
            self.flagged_tiles.append(self.dp)
            self.index += 1
            self.dp = self.get_next()
        elif b.description == 'Exit':
            self.index = self.count

        # TODO should call next i think...
        self.clean()

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
                #ax.legend()


    def _display_image(self):
        plt.figure(figsize = (7, 7))
        ax = plt.subplot()
        ax.axis("off")
        img = np.rollaxis(self.dp.image.compute(),0,3)
        ax.imshow(img)
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

        #dp = self.get_next()
        if self.dp is not None and self.index != self.count:
            print("%0.f tiles out of %0.f tiles have been cleaned" %
                 (self.index, self.count))
            self._display_image()
            self._display_polygons()
            display(HBox(buttons))
        else:
            print('all tiles have been cleaned')
            ## TOOD: add validation function for flagged tiles here
