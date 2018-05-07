class Model(object):
    """ Methods for accessing training data pairs """
    def __init__(self, item, shape=(3,256,256), dtype="uint8"):
        self.conn = requests.Session()
        self.conn.headers.update( headers )
        self.data = item["data"]
        self.links = item["links"]
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def id(self):
        return self.data["_id"]

    def save(self, data):
        return self.conn.put(self.links["update"]["href"], json=data).json()

    def update(self, new_data, save=True):
        self.data.update(new_data)
        if save:
            self.save(new_data)

    def remove(self):
        return self.conn.delete(self.links["delete"]["href"]).json()

    def __repr__(self):
        return json.dumps(self.data)
