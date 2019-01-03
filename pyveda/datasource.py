from pyveda.vedaset import VedaBase, VedaStream

''' basic use v.3:

    # connect to veda to stream data
    with DataSource.connect('Austin Buildings') as ds:
        ds.train.labels #etc

    # connect to veda and save locally to h5
    ds = DataSource.connect('Austin Buildings')
    ds.save('austin.h5')

    # open the h5 at a later date
    with DataSource.open('austin.h5) as ds:
        ds.train.labels #etc '''


class DataSource():
    ''' DataSources provide analysis-ready data to ML training.
        They can connect to a collection in Veda and stream data.
        They can save a connected collection to a local file.
        They can also load a locally saved collection. '''

    # TODO: move over the functionality from VedaCollectionProxy
    #       needed here and the API stuff over to Collection and Sample       

    def __init__(self, source=None):
        ''' Create a new DataSource

        Args:
            source: a VedaBase or VedaStream
        Returns:
            DataSource'''

        self.source = source

    @classmethod
    def open(cls, path):
        ''' Opens a local Vedabase for reading 

        Args:
            path(str): path to vedabase H5 file
        
        Returns:
            DataSource backed by VedaBase'''
        vedabase = VedaBase.from_path(path)
        datasource = cls(vedabase)
        return datasource
    
    @classmethod
    def connect(cls, identifier):
        ''' Connects to a remote Veda Collection for streaming

        Args:
            identifier(str): name or id of collection
        
        Returns:
            DataSource backed by VedaStream'''

        # TODO move over the VC stuff once the proxy is split up

        vedastream = VedaStream.from_vc()
        datasource = cls(vedastream)
        return datasource

    def save(self, path, count=None, partition=None):
        ''' Saves a remote Veda collection to a local H5 Vedabase

        Args:
            path(str): path to save the H5 to
        
        Returns:
            DataSource backed by VedaBase'''
        # should you be able to save if you're already a vedabase?
        if type(self.source) == VedaBase:
            raise NotImplementedError('Only remote collections opened with connect() can save to local files') 
        src = self.source
        vedabase = VedaBase.from_path(filename,
                          mltype=src.mltype,
                          klasses=src.classes,
                          image_shape=src.imshape,
                          image_dtype=src.dtype,
                          **kwargs)
        if count is None:
            count = src.count
        if partition is None:
            # not ideal since it hides the fallback behind None
            partition = src.partition or [70,20,10]
        urlgen = src.gen_sample_ids(count=count)
        token = gbdx.gbdx_connection.access_token
        build_vedabase(vedabase, urlgen, partition, count, token,
                        label_threads=1, image_threads=10, **kwargs)
        vedabase.flush()
        # should this update it's source?
        # or return a new datasource?
        # or return true for success?
        self.source = vb
        return self
        
    @property
    def test(self):
        ''' return the source testing group '''
        return self.source.test

    @property
    def train(self):
        ''' return the source training group '''
        return self.source.train

    @property
    def validate(self):
        ''' return the source validation group '''
        return self.source.validate 

    # needs to be added for vedabases?
    @property
    def count(self):
        return self.source.count

    def __len__(self):
        return sum([len(self.train), len(self.test), len(self.validate)])

    def __enter__(self):
        self.source.__enter__()
        return self

    def __exit__(self, *args):
        self.source.__exit__(*args)

    def __repr__(self):
        return "DataSource using {}".format(self.source.__repr__())

    def __del__(self):
        return self.source.__del__()

    def __getitem__(self, slc):
        return self.source.__getitem__()
    