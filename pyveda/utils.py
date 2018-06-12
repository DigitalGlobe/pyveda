import json
import datetime
import os
import uuid
import tempfile
import shutil

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

class NamedTemporaryHDF5File(object):
    def __init__(self, prefix="veda", suffix="h5", path=None, delete=True):
        self.name = mktempfilename(prefix, suffix=suffix, path=path)
        self._delete = delete
        self._fh = h5py.File(self.name, "a")

    def __enter__(self):
        return self._fh

    def __exit__(self):
        if self._delete:
            self.remove()

    def remove(self):
        try:
            self._fh.close()
        except Exception:
            pass
        try:
            os.remove(self.name)
        except Exception:
            pass

class NamedTemporaryHDF5Generator(object):
    def __init__(self, dirpath=None, delete=True, delete_files=True):
        if not dirpath:
            dirpath = tempfile.mkdtemp()
        self.dirpath = dirpath
        self._delete = delete
        self._delete_files = delete_files
        self._fps = []

    def __enter__(self):
        return self

    def __exit__(self):
        if self._delete_files:
            for h5 in self._fps:
                h5.remove()
        if self._delete:
            self.remove()

    def remove(self):
        for h5 in self._fps:
            h5.remove()
        shutil.rmtree(self.dirpath)

    def mktemp(self, prefix="veda", suffix="h5", delete=True):
        h5 = NamedTemporaryHDF5File(prefix=prefix, suffix=suffix,
                                                path=self.dirpath, delete=delete)
        self._fps.append(h5)
        return h5._fh, h5.name

    def mktempfilename(self, prefix="veda", suffix="h5"):
        return mktempfilename(prefix, suffix=suffix, path=self.dirpath)



def extract_load_tasks(dsk):
    """
    Given a veda image dask, extract request data into
    (url, token) tuple
    """
    d = dict(dsk)
    for k in d:
        if isinstance(k, str) and k.startswith("load_image"):
            task = d[k]
            url = task[2][0]
            token = task[2][1]
            return (url, token)
    return None

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
