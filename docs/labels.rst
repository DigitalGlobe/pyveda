Features and Labels
######################

Veda features are serialized as geojson features. The machine learning labels are specified in the ``Properties`` item of the geojson feature. The label format varies based on the machine learning type.

Label Formats
===============

Classification
----------------

**Input**

Geojson features must have a **single** positive class , from either a ``Properties`` field called ``label``, mapped from another field using the ``label_field`` parameter, or supplied as a default value using the ``default_value`` parameter. All of these examples add a positive label of the class _house_ to the matching image. 

Classification labels are ``1`` for positive detection and ``0`` for the absence of the class. It is not needed to have negative classes labelled - if a class is missing it will automatically be labelled as ``0``. A shorthand version passing just the positive class name is also acceptable. The class name does not have to be a string as an input, but will be stored as a string. 

A full label of a ``house`` feature, in a VedaCollection that has classes ``['house', 'car', 'boat']`` is:

.. code-block:: javascript

    'Properties': {
        'label': {'house':1, 'car':0, 'boat':0}
    }


The same feature with a label with the positive value only:


.. code-block:: javascript

    'Properties': {
        'label': {'house':1}
    }


Shorthand version:

.. code-block:: javascript

    'Properties': {
        'label': 'house'
    }


Using the parameter ``label_field``:

.. code-block:: javascript

    // using label_field='building_type'
    'Properties': {
        'building_type': 'house'
    }


or ``default_value``:

.. code-block:: python

    default_value='house'




The parent VedaCollection can determine the classes on the first load of data, or the classes can be set by passing an optional `classes` parameter on initialization, in the case that not all classes are present in the first set of data uploaded. Classes are fixed and can not be modified after the first set of data is loaded.

Classification features can be any geometry type, though it is recommended that multi-geometry types be exploded to individual features.



**DataPoint Output**

The DataPoint class has a ``labels`` property that includes all labels in the parent VedaCollection. This is a dictionary keyed with class names, with 0 or 1 labels that indicate positive or negative presence of that class in the image tile. For this and all following examples we'll assume the image tile has houses and cars, but no boats. The DataPoint API JSON response has a labels property in the ``Properties`` field.

.. code-block:: python

    datapoint.labels = {
        'house': 1,
        'car': 1,
        'boat': 0
    }




**VedaBase Output**

The VedaBase class stores labels as a NumPy array of boolean values using one-hot encoding. The label classes are stored in a `classes` property that maps to the same position in the label array. The label order is always alphabetical.

.. code-block:: python

    vedabase.classes = ['boat', 'car', 'house']
    vedabase.train.labels[0] = [0, 1, 1]




Object Detection
------------------

**Input**

Input features for object detection share the same label requirements as for classification. The objects will be stored as the bounding box of the feature's geometry. The geometry must be of type Polygon but does not need to be a rectangle. Point and Polyline geometries will need to be buffered by an appropriate amount first.

**DataPoint Output**

The `label` property of the DataPoint is similar in structure to the Classification case, with keys representing  classes. The value of each class is a list of features representing the bounding box of the input geometries. The features use GeoJSON structure, but use the NumPy convention of the origin at top left, and uses units of pixels.

.. code-block:: python

    datapoint.labels[0] = {
        'house': [f1, f2,..fn],
        'car': [f1, f2,..fn],
        'boat': []
    }




** VedaBase Output**


Object detection classes are stored in the `classes` property in the same manner as Classification data. Because object bounding boxes can overlap, the labels are in the form of a list of lists of bounding boxes. The bounding boxes are lists of `[minx, miny, maxx, maxy]` in pixel coordinates with the origin at top left. The position of the classes and labels lists match.

.. code-block:: python

    vedabase.classes = ['boat', 'car', 'house']
    vedabase.train.labels[0] = [
        [], # boat bboxes
        [[0, 0, 1, 1], [1, 2, 4, 5], [3, 3, 7, 9]], # car bboxes
        [[0, 0, 1, 1], [1, 2, 4, 5], [3, 3, 7, 9]] # house bboxes
    ]






Segmentation
----------------

**Input**

The requirements for input data features for segmentation are the same as for object detection. The object must be a Polygon. The geometry will automatically be clipped to fit inside the image tile.

**DataPoint Output**

The `label` property of the DataPoint has the same structure as for object detection, except the label features represent the input feature's geometry instead of the feature's bounding box. The coordinate values are again in pixels from the top left of the image.

**VedaBase Output**

Segmentation classes continue in the same structure as the other machine learning types, but include a ``None`` value in the first position representing the pixels that have no segmentation data. The background pixel value is set to 0, and the segmented pixels store values that match the array index of the class name in the ``classes`` list. The ``labels`` for segmentation are 2D NumPy arrays with the pixels representing the class list indices. The 2D array is the same size as the image tile.

.. code-block:: python

    vedabase.classes = [None, 'boat', 'car', 'house']
    vedabase.train.labels[0] = [
        [0, 0, 0, 0, 0, ...], 
        [0, 3, 3, 0, 2, ...],
        [0, 3, 3, 0, 0, ...]
    ]



