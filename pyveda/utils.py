import json

def rda(dsk):
    return [json.dumps({
      'graph': dsk.ipe_id,
      'node': dsk.ipe.graph()['nodes'][0]['id'],
      'bounds': dsk.bounds
    })]

def maps_api(dsk):
    return [json.dumps({
      'bounds': dsk.bounds
    })]


def transforms(source):
    return rda if source == 'rda' else maps_api
