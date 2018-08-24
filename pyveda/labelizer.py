import ipywidgets as widgets
from ipywidgets import Button, HBox, VBox
from IPython.display import display, clear_output
from rasterio import features
from shapely.geometry.geo import shape
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gbdxtools import Interface, TmsImage, CatalogImage,IpeImage
from rasterio import features
from shapely.geometry.geo import shape

def format_data(chips, polygons):
    """formats chip/polygon data for object vote"""
    formatted_chips = []
    formatted_polygons = []
    for i,a in enumerate(chips):
        polys = polygons[i]
        for b in polys:
            formatted_chips.append(a)
            formatted_polygons.append(b)
    return(formatted_chips, formatted_polygons)

class voting(object):
    """Methods for data generation and verification"""

    def __init__(self, chips, class_type, polygons=None):
        self.chips = chips
        self.polygons = polygons
        self.class_type = class_type
        self.index = 0
        self.samples = []
        self.positive_samples = []
        self.negative_samples = []

    def _create_buttons(self):
        """
        Creates Ipython widget buttons
        """
        buttons = []
        actions = [('Yes', 'success'), ('No', 'danger'), ('Back', 'danger'),
                   ('Skip', 'warning'), ('Exit', 'info')]
        for b in actions:
            btn = Button(description = b[0], button_style = b[1])
            buttons.append(btn)
        return buttons

    def _handle_buttons_binary(self, b):
        """
        Callback for the widget buttons.
        Appends an image chip and associated binary classification to a list,
        then re-generates the widget.
        """
        if b.description == 'Yes':
            self.samples.append((self.chips[self.index], [1]))
            self.index += 1
        elif b.description == 'No':
            self.samples.append((self.chips[self.index], [0]))
            self.index += 1
        elif b.description == 'Back':
            self.samples.pop()
            self.index -= 1
        elif b.description == 'Exit':
            self.index = len(self.chips)
        self.binary_vote()

    def binary_vote(self):
        """
        Method for classifying/voting on binary data.
        """
        clear_output()
        buttons = self._create_buttons()
        for b in buttons:
            b.on_click(self._handle_buttons_binary)
        if self.index < len(self.chips):
            print("%0.f chips out of %0.f chips have been labeled" % (self.index, len(self.chips)))
            display(HBox(buttons))
            plt.figure(figsize = (7, 7))
            plt.axis("off")
            plt.imshow(self.chips[self.index].rgb(bands = [4, 2, 1]))
            plt.title("Is there a %s in this chip?" % self.class_type)
        if self.index >= len(self.chips):
            print('all objects have been labeled')
        return(self.samples)

    def _handle_buttons_object(self, b):
        """Not currently in love with this either, I'm not crazy about two different outputs.
           That being said, I am not sure what else to do--simply trash the negative samples?
           Use one list and add a binary component? Use one list and return negative samples with no polygons?
           If sticking with two outputs, how to handle 'back'?"""

        if b.description == 'Yes':
            print("positive sample")
            self.positive_samples.append((self.chips[self.index], self.polygons[self.index]))
            self.index += 1
        elif b.description == 'No':
            print("negative sample")
            self.negative_samples.append((self.chips[self.index], self.polygons[self.index]))
            self.index += 1
        elif b.description == 'Exit':
            self.index = len(self.polygons)
            print("bye")

        self.object_vote()

    def _plot_polygons(self):
        """polygon plotter for object_vote"""
        ax = plt.subplot()
        if np.size(self.polygons[self.index]) == 1:
            pxb=self.chips[self.index].pxbounds(self.polygons[self.index])
            ax.add_patch(patches.Rectangle((pxb[0], pxb[1]), (pxb[2] - pxb[0]),\
                    (pxb[3] - pxb[1]), edgecolor = 'red', fill = False, lw=2))

    def object_vote(self):
        """method for varifying segmentation or object detection data"""
        clear_output()
        buttons = self._create_buttons()
        for b in buttons:
            b.on_click(self._handle_buttons_object)
        if self.index < len(self.polygons):
            print("%0.f  out of %0.f objects have been labeled" % (self.index, len(self.polygons)))
            display(HBox(buttons))
            plt.figure(figsize=(7,7))
            ax=plt.subplot()
            ax.axis("off")
            ax.imshow(self.chips[self.index].rgb(bands=[4, 2, 1]))
            self._plot_polygons()
            plt.title('Is this a %s?' % self.class_type)
        if self.index >= len(self.polygons):
            print('all objects have been labeled')
        return(self.positive_samples, self.negative_samples)
