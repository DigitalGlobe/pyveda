import six

if six.PY3:
    from pyveda.fetch.compat.fetchpy3 import write_fetch
if six.PY2:
    from pyveda.fetch.compat.fetchpy2 import write_fetch
