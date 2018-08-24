import ipywidgets as widgets
from ipywidgets import Button, HBox, VBox
from IPython.display import display, clear_output
from rasterio import features
from shapely.geometry.geo import shape
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gbdxtools import Interface, TmsImage, CatalogImage,IpeImage

class voting(object):
    """Methods for data generation and verification"""

    def __init__(self, chips, class_type, polygons=None):
        self.chips = chips
        self.polygons = polygons
        self.class_type = class_type
        self.index = 0
        self.samples = []

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

    def _handle_buttons(b):
        """
        Callback for the widget buttons.
        Appends an image chip and associated binary classification to a list,
        then re-generates the widget.

        Note: Having problems here. I don't understand why the variables
        initialized with the class won't pass into this method. 
        """
        if b.description == 'Yes':
            self.samples.append((self.chips[self.index], [1]))
            self.index += 1
        elif b.description == 'No':
            self.samples.append((self.chips[index], [0]))
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
        buttons = voting._create_buttons(self)
        for b in buttons:
            b.on_click(voting._handle_buttons)
        if self.index < len(self.chips):
            print("%0.f chips out of %0.f chips have been labeled" % (self.index, len(self.chips)))
            display(HBox(buttons))
            plt.figure(figsize = (7, 7))
            plt.axis("off")
            plt.imshow(self.chips[self.index].rgb(bands = [4, 2, 1]))
            plt.title("Is there a %s in this chip?" % self.class_type)
        if self.index >= len(self.chips):
            print('all objects have been labeled')
        return(samples)
