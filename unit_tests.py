import pytest

r = pytest.main(["-s", "pyveda/tests/unit"])
if r:
    raise Exception("There were test failures or errors.")
