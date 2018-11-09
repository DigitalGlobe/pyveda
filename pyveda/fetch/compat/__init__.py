import sys

if sys.version_info[0:2] >= (3,4):
    from pyveda.fetch.compat.fetchpy3 import build_vedabase
else:
#    from pyveda.fetch.compat.fetchpy2 import write_fetch
    raise ImportError("VedaBase usage requires python > 3.4")
