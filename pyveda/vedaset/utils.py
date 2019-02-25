from functools import partial
import tables
from pyveda.utils import ignore_warnings

ignore_NaturalNameWarning = partial(ignore_warnings,
                                    _warning=tables.NaturalNameWarning)

def extend_instance(obj, cls):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (cls, base_cls),{})

