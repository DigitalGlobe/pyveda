import json
import datetime
import os
import uuid

def mklogfilename(prefix, suffix="json", path=None):
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    basename = "_".join([prefix, timestamp]) # e.g. 'mylogfile_120508_171442'
    filename = ".".join([basename, suffix])
    if path:
        filename = os.path.join(path, filename)
    return filename

def mktempfilename(prefix, suffix="h5", path=None):
    idstr = str(uuid.uuid4())
    basename = "_".join([prefix, idstr])
    filename = ".".join([basename, suffix])
    if path:
        filename = os.path.join(path, filename)
    return filename

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
