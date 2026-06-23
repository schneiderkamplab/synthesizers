from .base import Model
from ..utils import quiet_synthcity_imports

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
        with quiet_synthcity_imports():
            from synthcity.plugins.core.plugin import Plugin
            self.model = Plugin.load(state)
    def dump_model_data(self):
        return self.model.save()
    def load_model_data(model_data):
        with quiet_synthcity_imports():
            from synthcity.plugins.core.plugin import Plugin
            model = Plugin.load(model_data)
        return SynthCityModel(model)
    def __repr__(self):
        return f"SynthCityModel({repr(self.model)})"
