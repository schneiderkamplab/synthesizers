from pickle import dumps, loads

from .base import Model

class SynthPopModel(Model):
    MODEL_FILE = "synthpop.xml"
    MODEL_FILE_PART = "synthpop_{0}.xml"
    MODEL_TYPE = "SynthPopModel"
    def __init__(self, model):
        super(SynthPopModel, self).__init__(saveable=True)
        self.model = model
    def dump_model_data(self):
        return dumps(self.model)
    def load_model_data(model_data):
        model = loads(model_data)
        return SynthPopModel(model)
    def __repr__(self):
        return f"SynthPopModel({repr(self.model)})"
