## Python Reference

### Create a new TraingingSet

```python

from pyveda import TrainingSet

td = TrainingSet('A Name for the Data', classes=['building', 'cars', 'planes'], source="maps_api", mlType="classification", bbox=[minx, miny, maxx, maxy])

mydata = [ ( CatalogImage, [0, 0, 1] ), ( CatalogImage, [1, 0, 1] ), ( CatalogImage, [0, 1, 0] )]

td.feed(mydata, group="train")

print( td.count )

# ----> { "train": 3 }

td.save()

```

### Search

```python
# find datasets
from pyveda import search
datasets = search( { search params } )
```

```python
data = datasets[0]

# Random set of data points...
x,y = data.batch(5)
print(x.shape, y.shape)

# Generators of random data...
for x,y in data.batch_generator(5):
    print(x.shape, y)

# Sequential Access...
pnts = data[0:10]
len(pnts)

# Persist chages to data...
pnt = pnts[0]
pnt.update({'y': [1]}, save=True)

```
