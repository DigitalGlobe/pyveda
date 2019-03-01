import os 
import tarfile
import tempfile
import mmap
import json
from pyveda.utils import url_to_numpy
from requests_toolbelt.multipart.encoder import MultipartEncoder
from pyveda.veda.api import DataSampleClient, VedaCollectionProxy
from pyveda.config import VedaConfig

cfg = VedaConfig()

def search(params={}):
    r = cfg.conn.post('{}/models/search'.format(cfg.host), json=params)
    try:
        results = r.json()
        return [Model.from_doc(s) for s in results]
    except Exception as err:
        print(err)
        return []

def create_archive(model, weights): 
    """ Creates a tar from a model """
    dirpath = tempfile.mkdtemp()
    name =  "{}/model.tar.gz".format(dirpath)
    print("Creating model archive: {}".format(name))
    with tarfile.open(name, "w:gz") as tar:
        if model is not None:
            tar.add(model, arcname='model.json')
        if weights is not None:
            tar.add(weights, arcname='weights.h5')
    return name
        

class Model(object):
    """ 
      Defines a Model object for saving and accessing models in Veda. 

      Args:
          name (str): a name for the model
          model_path (str): path to the serialized model file  
          weights_path (str): path to the serialized model weights 
          archive (str): a path to a local tar.gz archive for the model 
          mltype (str): the mltype of the model
          bounds (list): a bounding box (minx, miny, maxx, maxy)
          imshape (tuple): the shape of the images the model expects 
          training_set (VedaCollection): a veda training data collection. Used as a reference 
                                         to the data the model was trained from. If provided, bounds, shape,
                                         dtype, and mltype will be inherited.
          library (str): The ml framework used for training the model (keras, pytorch, tensorflow)
          classes (list): A list of classes that the model should return

    """
    def __init__(self, name, model_path=None, weights_path=None, mltype=None, channels_last=False, **kwargs):
        self.id = kwargs.get("id", None)
        self.links = kwargs.get("links")
        self.channels_last = channels_last
        self.meta = self._construct_meta(name, mltype=mltype, **kwargs)
        for k,v in self.meta.items():
            setattr(self, k, v)

        assert self.mltype is not None, "Must define an mltype as one of `classification`, `object_detection`, or `segmentation`"
        assert self.library is not None, "Must define `library` as one of `keras`, `pytorch`, or `tensorflow`"
           
        if 'archive' in kwargs: 
            self.archive = kwargs.get('archive')
        elif model_path is not None or weights_path is not None:
            self.archive = create_archive(model_path, weights_path)

    @classmethod
    def from_doc(cls, doc):
        """ Helper method that converts a db doc to a Model """
        return cls(**doc['properties'])

    @classmethod
    def from_id(cls, _id):
        """ Helper method that fetches an id into a model """
        url = "{}/models/{}".format(cfg.host, _id)
        r = cfg.conn.get(url)
        r.raise_for_status()
        return cls.from_doc(r.json())

    def save(self):
        payload = MultipartEncoder(
            fields={
                'metadata': json.dumps(self.meta),
                'model': (os.path.basename(self.archive), open(self.archive, 'rb'), 'application/octet-stream')
            }
        )

        if self.links is not None:
            url = self.links['self']['href']
        else:
            url = "{}/models".format(cfg.host)
        r = cfg.conn.post(url, data=payload, headers={'Content-Type': payload.content_type})
        r.raise_for_status()
        doc = r.json()
        self.id = doc["properties"]["id"]
        self.links = doc["properties"]["links"]
        del doc["properties"]["links"]
        self.meta.update(doc['properties'])
        self.refresh(meta=self.meta)
        return self

    def deploy(self):
        # fetch the latest model data from the server, need to make sure we've saved the tarball
        self.refresh()
        assert self.id is not None, "Model not saved, please call save() before deploying."
        assert self.library is not None, "Model library not defined. Please set the `.library` property before deploying."
        assert self.meta["location"] is not None, "Model not finished saving yet, model.location is None..."
        if self.deployed is None or self.deployed["id"] is None:
            cfg.conn.post(self.links["deploy"]["href"], json={"id": self.id})
            self.refresh()
            return self.deployed 
        else:
            print('Model already deployed.')
    
    def predict(self, bounds, image=None, **kwargs):
        ''' 
          Run predictions for an AOI within an RDA image based image. 

          Args:
            bounds (list): bounding box AOI
            image (RDAImage): An ERDA based image to use for streaming tiles
        '''
        assert self.deployed is not None, "Model no deployed, please call deploy() before running predictions"
        if image is None: 
            rda_id = kwargs.get('rda_id')
            rda_node = kwargs.get('rda_node')
        else:
            rda_id = image.rda_id
            rda_node = image.rda.graph()['nodes'][0]['id']
        assert rda_id is not None, "Must at least provide and image object or an rda_id"

        meta = {
            "name": kwargs.get('name', self.name),
            "description": kwargs.get("description", None),
            "dtype": self.dtype, 
            "mltype": self.mltype,
            "imshape": kwargs.get('imshape', self.imshape),
            "public": False,
            "bounds": bounds,
            "classes": self.classes,
            "channels_last": str(self.channels_last), 
            "deployed_model": self.deployed['id']
        }
        if self.channels_last == 'true':
            meta["imshape"] = self.imshape[::-1]
        payload = {
            "id": self.id,
            "metadata": meta,
            "options": {
              "graph": rda_id,
              "node": rda_node
            }
        }
        doc = cfg.conn.post(self.links["predict"]["href"], json=payload).json()
        return PredictionSet.from_doc(doc)

        
    def update(self, new_data, save=True):
        self.meta.update(new_data)
        if save:
            return cfg.conn.put(self.links["update"]["href"], json=self.meta).json()

    def remove(self):
        self.id = None
        cfg.conn.delete(self.links["delete"]["href"])

    def publish(self):
        assert self.id is not None, 'You can only publish a saved Model. Call the save method first.'
        return cfg.conn.put(self.links["update"]["href"], json={"public": True}).json()

    def unpublish(self):
        assert self.id is not None, 'You can only unpublish a saved Model. Call the save method first.'
        return cfg.conn.put(self.links["update"]["href"], json={"public": False}).json()

    def download(self, path=None):
        assert self.id is not None, 'You can only download a saved Model. Call the save method first.'
        path = path if path is not None else './{}'.format(self.id)
        try:
            os.makedirs(path)
        except Exception as err:
            pass
        r = cfg.conn.get(self.links["download"]["href"])
        with open('{}/model.tar.gz'.format(path), 'wb') as fh:
            fh.write(r.content)
        return path

    def refresh(self, meta=None):
        if meta is None:
            r = cfg.conn.get(self.links["self"]["href"])
            r.raise_for_status()
            meta = {k: v for k, v in r.json()['properties'].items()}
        self.meta.update(meta)
        for k,v in self.meta.items():
            setattr(self, k, v)
    
    def _construct_meta(self, name, **kwargs):
        has_vcp = False
        vcp = kwargs.get('training_set')
        override_vals = ["bounds", "imshape", "dtype", "classes", "mltype"]
        if vcp and not isinstance(vcp, str):
            vcp_id = vcp.id
            overrides = {v:getattr(vcp,v) for v in override_vals}
            if 'dtype' in overrides:
                overrides['dtype'] = overrides['dtype'].name
        else:
            vcp_id = vcp
            overrides = {}
  
        # override any values from VCP that may be in from kwargs 
        for v in override_vals:
            if v in kwargs:
                overrides[v] = kwargs.get(v)

        meta = {
          "name": name,
          "deployed": kwargs.get("deployed", {"id": None}),
          "description": kwargs.get("description", None),
          "public": kwargs.get("public", False),
          "library": kwargs.get("library", {}),
          "location": kwargs.get("location", {}),
          "training_set": vcp_id,
          "channels_last": self.channels_last
        }
        meta.update(overrides)
        return meta


    def __repr__(self):
        return json.dumps(self.meta)


class PredictionSampleClient(DataSampleClient):
  @property
  def prediction(self):
      url = "{}/predictions/result/{}".format(cfg.host, self.id)
      return url_to_numpy(url, cfg.conn.access_token)

class PredictionSet(VedaCollectionProxy):
    """ Methods for accessing training data pairs """
    _data_sample_client = PredictionSampleClient

    def __init__(self, **kwargs):
        self._meta = {k: v for k, v in kwargs.items() if k in self._metaprops}
        self._dataset_base_furl = "{host_url}/predictions/{dataset_id}"
        self.id = kwargs.get('id')
        self._dataset_id = self.id

    @classmethod
    def from_doc(cls, doc):
        """ Helper method that converts a db doc to a PredictionSet """
        return cls(**doc['properties'])

    @classmethod
    def from_id(cls, _id):
        """ Helper method that fetches an id into a predictionset """
        url = "{}/predictions/{}".format(cfg.host, _id)
        r = cfg.conn.get(url)
        r.raise_for_status()
        return cls.from_doc(r.json())

    @classmethod
    def search(cls, params={}):
        r = cfg.conn.post('{}/predictions/search'.format(cfg.host), json=params)
        try:
            results = r.json()
            return [cls.from_doc(s) for s in results]
        except Exception as err:
            return []

