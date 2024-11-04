from synthcity.plugins import Plugins

from .base import Model

class SynthCityModel(Model):
    MODEL_FILE = "synthcity.xml"
    MODEL_FILE_PART = "synthcity_{0}.xml"
    MODEL_TYPE = "SynthCityModel"
    def __init__(self, model):
        super(SynthCityModel, self).__init__(saveable=True)
        self.model = model
    def __getstate__(self):
        return self.model.save()
    def __setstate__(self, state):
        self.model = Plugins().load(state)
    def dump_model_data(self):
        return self.model.save()
    def load_model_data(model_data):
        model = Plugins().load(model_data)
        return SynthCityModel(model)
    def __repr__(self):
        return f"SynthCityModel({repr(self.model)})"
