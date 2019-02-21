classification_item = {
    'properties': {
        'label': {
            'house':1, 'car':1, 'boat':0
            }
    }
}

segmentation_item = {
    "properties": {
        "label": {
            "building": [
                {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [52, 84],
                            [9, 195],
                            [0, 102],
                            [0, 69],
                            [52, 84]
                        ]
                    ]
                }
            ]
        }
    }
}

objd_item = {
    "properties": {
        "label": {
            "damaged building": [],
            "building": [
                [
                    235,
                    62,
                    256,
                    117
                ]
            ]
        }
    }
}