import ipywidgets as widgets
from ipywidgets import Button, HBox, VBox
from IPython.display import display, clear_output
from rasterio import features
from shapely.geometry.geo import shape
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rasterio import features
from shapely.geometry.geo import shape


def format_data(chips, polygons):
    """
    formats chip/polygon data for use in the voting method object_vote.
    args:
        chips: A list of image chips/tiles
        polygons: A list of polygons associated with each image tile
    returns:
        formatted_chips: Chips formatted for use in object_vote
        formatted_polygons: Polygons formatted for use in object_vote
    """
    formatted_chips = []
    formatted_polygons = []
    for i, a in enumerate(chips):
        polys = polygons[i]
        for b in polys:
            formatted_chips.append(a)
            formatted_polygons.append(b)
    return(formatted_chips, formatted_polygons)


class voting(object):
    """
    Methods for data generation and verification
    Args:
        chips: a list of image tiles as dask arrays
        class_type: the type of object to be verified or classified
    Params:
        polygons: a list of label geometries associated with chips
    """

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
        Creates ipywidget widget buttons for use in object_vote() and
        binary_vote().
        Returns:
            buttons: A list of ipywidget Button() objects
        """
        buttons = []
        actions = [('Yes', 'success'), ('No', 'danger'), ('Back', 'info'),
                   ('Skip', 'info'), ('Exit', 'info')]
        for b in actions:
            btn = Button(description=b[0], button_style=b[1])
            buttons.append(btn)
        return buttons

    def _handle_buttons_binary(self, b):
        """
        Callback and handling of widget buttons for binary_vote().
        Appends an image chip and associated binary classification (tuple) to
        samples, then re-generates the widget.
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
        Method for classifying/voting on binary image data.
        Displays an image chip and associated ipywidget buttons. Allows user to
        classify tiles as positive samples (containing an object) or negative
        samples (without an object).
        Returns:
            samples: A list of tuples. The first value is the image chip, the
            second value is the binary classification ([0] or [1]).
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
        """
        Callback and handling of widget buttons for object_vote().
        Appends an image chip and associated label geometry to either
        positive_samples or negative_samples.
        """
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
        """Adds a label geometry to the image tile plot."""
        ax = plt.subplot()
        if np.size(self.polygons[self.index]) == 1:
            pxb = self.chips[self.index].pxbounds(self.polygons[self.index])
            ax.add_patch(patches.Rectangle(
                (pxb[0], pxb[1]), (pxb[2] - pxb[0]),
                (pxb[3] - pxb[1]), edgecolor = 'red', fill = False, lw = 2))

    def object_vote(self):
        """
        Method for verifying image data with associated polygons.
        Displays a polygon overlayed on image chip with associated ipywidget
        buttons. Allows user to classify each polygon as a positive sample
        (correctly labeled) or a negative sample (incorrectly labled).
        Returns:
            positive_samples: A list of tuples. The first value is the image
            chip, the second value is the associated label geometry.
            positive_samples: A list of tuples. The first value is the image
            chip, the second value is the associated label geometry.
        """
        clear_output()
        buttons = self._create_buttons()
        for b in buttons:
            b.on_click(self._handle_buttons_object)
        if self.index < len(self.polygons):
            print("%0.f  out of %0.f objects have been labeled" % (self.index, len(self.polygons)))
            display(HBox(buttons))
            plt.figure(figsize = (7, 7))
            ax = plt.subplot()
            ax.axis("off")
            ax.imshow(self.chips[self.index].rgb(bands = [4, 2, 1]))
            self._plot_polygons()
            plt.title('Is this a %s?' % self.class_type)
        if self.index >= len(self.polygons):
            print('all objects have been labeled')
        return(self.positive_samples, self.negative_samples)
