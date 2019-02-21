import os
import pycurl
import tables
import numpy as np
import ujson as json
from skimage.io import imread
from tempfile import NamedTemporaryFile
from pyveda.utils import mklogfilename


def write_trace_profile(fname, nreqs, trace_cache):
    basepath, inputfile = os.path.split(fname)
    basename = "_".join([inputfile.split(".")[0], "n{}".format(nreqs)])
    filename = mklogfilename(basename, suffix="json", path=basepath)
    with open(filename, "w") as f:
        json.dump(trace_cache, f)
    return filename


def url_to_array(url, token):
    _curl = pycurl.Curl()
    _curl.setopt(_curl.URL, url)
    _curl.setopt(pycurl.NOSIGNAL, 1)
    _curl.setopt(pycurl.HTTPHEADER,
                 ['Authorization: Bearer {}'.format(token)])

    with NamedTemporaryFile(prefix="veda",
                            suffix=".tif", delete=False) as temp:
      try:
          _curl.setopt(_curl.WRITEDATA, temp.file)
          _curl.perform()
          code = _curl.getinfo(pycurl.HTTP_CODE)
          if code != 200:
             raise TypeError(
                 "Request for {} returned error code: {}".format(url, code))
          temp.file.flush()
          temp.close()
          _curl.close()
          return imread(temp.name)
      except Exception as err:
          print('Error fetching image...', err)


def _atom_from_dtype(_type):
    if isinstance(_type, np.dtype):
        return tables.Atom.from_dtype(_type)
    return tables.Atom.from_dtype(np.dtype(_type))


