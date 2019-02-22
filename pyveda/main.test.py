from pyveda.main import create_from_geojson

def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 5
