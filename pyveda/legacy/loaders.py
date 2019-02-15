import numpy as np
import dask.array as da
from dask import delayed
import random
import json

import os
from collections import defaultdict
import threading
from tempfile import NamedTemporaryFile
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

try:
    from functools import lru_cache  # python 3
except ImportError:
    from cachetools.func import lru_cache

from skimage.io import imread
import pycurl


MAX_RETRIES = 2
_curl_pool = defaultdict(pycurl.Curl)


@delayed
def load_image(url, token, shape, dtype=np.float32):
    """ Loads a geotiff url inside a thread and returns as an ndarray """
    _, ext = os.path.splitext(urlparse(url).path)
    success = False
    for i in range(MAX_RETRIES):
        thread_id = threading.current_thread().ident
        _curl = _curl_pool[thread_id]
        _curl.setopt(_curl.URL, url)
        _curl.setopt(pycurl.NOSIGNAL, 1)
        _curl.setopt(pycurl.HTTPHEADER, [
                     'Authorization: Bearer {}'.format(token)])
        # TODO: apply correct file extension
        with NamedTemporaryFile(prefix="sandman", suffix=ext, delete=False) as temp:
            _curl.setopt(_curl.WRITEDATA, temp.file)
            _curl.perform()
            code = _curl.getinfo(pycurl.HTTP_CODE)
            try:
                if(code != 200):
                    raise TypeError(
                        "Request for {} returned unexpected error code: {}".format(url, code))
                temp.file.flush()
                temp.close()
                arr = imread(temp.name)
                if len(arr.shape) == 3:
                    arr = np.rollaxis(arr, 2, 0)
                else:
                    arr = np.expand_dims(arr, axis=0)
                success = True
                return arr
            except Exception as e:
                _curl.close()
                del _curl_pool[thread_id]
            finally:
                temp.close()
                os.remove(temp.name)

    if success is False:
        arr = np.zeros(shape, dtype=dtype)
    return arr
