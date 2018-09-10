import requests
import os 
import mmap
import json
from gbdxtools import Interface
gbdx = Interface()

HOST = os.environ.get('SANDMAN_API')
if not HOST:
    HOST = "https://veda-api.geobigdata.io"

if 'https:' in HOST:
    conn = gbdx.gbdx_connection
else:
    headers = {"Authorization": "Bearer {}".format(gbdx.gbdx_connection.access_token)}
    conn = requests.Session()
    conn.headers.update(headers)

def search(params={}):
    r = conn.post('{}/models/search'.format(HOST), json=params)
    try:
        results = r.json()
        return [Model.from_doc(s) for s in results]
    except:
        return []


class Model(object):
    """ Methods for accessing training data pairs """
    def __init__(self, name, model_path=None, mlType="classification", bbox=[], shape=(3,256,256), dtype="uint8", **kwargs):
        self.id = kwargs.get('id', None)
        self.links = kwargs.get('links')
        self.shape = tuple(shape)
        self.dtype = dtype
        self.files = {
            "model": model_path, 
        }

        self.meta = {
            "name": name,
            "bbox": bbox,
            "mlType": mlType,
            "public": kwargs.get("public", False),
            "training_set": kwargs.get("training_set", None),
            "description": kwargs.get("description", None),
            "deployed": json.loads(kwargs.get("deployed", False)),
            "library": kwargs.get("library", None)
        }

        for k,v in self.meta.items():
            setattr(self, k, v)

    @classmethod
    def from_doc(cls, doc):
        """ Helper method that converts a db doc to a Model """
        doc['data']['links'] = doc['links']
        return cls(**doc['data'])

    @classmethod
    def from_id(cls, _id):
        """ Helper method that fetches an id into a model """
        url = "{}/models/{}".format(HOST, _id)
        r = conn.get(url)
        r.raise_for_status()
        return cls.from_doc(r.json())

    def save(self):
        files = self.files
        payload = {
            'metadata': (None, json.dumps(self.meta), 'application/json'),
            'model': (os.path.basename(files["model"]), open(files["model"], 'rb'), 'application/octet-stream')
        }

        if self.links is not None:
            url = self.links['self']['href']
            meta.update({"update": True})
            files["metadata"] = (None, json.dumps(meta), 'application/json')
        else:
            url = "{}/models".format(HOST)
        r = conn.post(url, files=payload)
        r.raise_for_status()
        doc = r.json()
        self.id = doc["data"]["id"]
        self.links = doc["links"]
        return doc

    def deploy(self):
        assert self.id is not None, "Model not saved, please call save() before deploying."
        assert self.library is not None, "Model library not defined. Please set the `.library` property before deploying."
        assert not self.deployed, "Model already deployed."
        return conn.post(self.links["deploy"]["href"], json={"id": self.id}).json()        
            
    def update(self, new_data, save=True):
        self.meta.update(new_data)
        if save:
            return conn.put(self.links["update"]["href"], json=self.meta).json()

    def remove(self):
        self.id = None
        conn.delete(self.links["delete"]["href"])

    def publish(self):
        assert self.id is not None, 'You can only publish a saved Model. Call the save method first.'
        return conn.put(self.links["publish"]["href"], json={"public": True}).json()

    def unpublish(self):
        assert self.id is not None, 'You can only unpublish a saved Model. Call the save method first.'
        return conn.put(self.links["publish"]["href"], json={"public": False}).json()

    def download(self, path=None):
        assert self.id is not None, 'You can only download a saved Model. Call the save method first.'
        path = path if path is not None else './{}'.format(self.id)
        try: 
            os.makedirs(path)
        except Exception as err:
            pass
        r = conn.get(self.links["download"]["href"])
        with open('{}/model.tar.gz'.format(path), 'wb') as fh:
            fh.write(r.content)
        return path
      

    def __repr__(self):
        return json.dumps(self.meta)
