from pyveda.utils import ignore_warnings
import tables

ignore_NaturalNameWarning = partial(ignore_warnings,
                                    _warning=tables.NaturalNameWarning)


