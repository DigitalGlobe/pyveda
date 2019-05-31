import os
import tables
import pickle
import numpy as np
import ujson as json
from functools import partial
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


def url_to_array(url, token, load_fn=imread, ext=".tif", chunksize=1024):
    headers = {"Authorization":"Bearer {}".format(token)}
    conn = requests.Session()
    conn.headers.update(headers)
    r = conn.get(url, stream=True)
    r.raise_for_status()
    with NamedTemporaryFile(prefix="veda", suffix=".tif") as temp:
        for chunk in r.iter_content(chunksize):
            temp.file.write(chunk)
        temp.file.flush()
        obj = load_fn(temp.name)
    return obj


url_to_numpy = partial(url_to_array, load_fn=np.load, ext=".npy")
url_unpickle = partial(url_to_array,
                       load_fn=lambda x: pickle.load(open(x, "rb")),
                       ext=".npy")


def _atom_from_dtype(_type):
    if isinstance(_type, np.dtype):
        return tables.Atom.from_dtype(_type)
    return tables.Atom.from_dtype(np.dtype(_type))


