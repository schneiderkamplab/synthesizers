from huggingface_hub import hf_hub_download
from pathlib import Path
from synthcity.plugins import Plugins

from .base import Model
from ..utils.xml import unwrap_model_xml, wrap_model_xml

MODEL_FILE = "synthcity.xml"
MODEL_TYPE = "SynthCityModel"

class SynthCityModel(Model):
    def __init__(self, model):
        super(SynthCityModel, self).__init__(saveable=True)
        self.model = model
    def __getstate__(self):
        return self.model.save()
    def __setstate__(self, state):
        self.model = Plugins().load(state)
    def save_pretrained(self, name):
        path = Path(name)
        path.mkdir(parents=True, exist_ok=True)
        model_data = self.model.save()
        xml_data = wrap_model_xml(model_type=MODEL_TYPE, model_data=model_data)
        with open(path / MODEL_FILE, "wb") as f:
            f.write(xml_data)
    def from_pretrained(name):
        path = Path(name)
        if not path.exists():
            hf_hub_download(repo_id=name, filename=MODEL_FILE)
        if not path.is_dir() or not (path / MODEL_FILE).is_file():
            raise RuntimeError(f"invalid synthcity model directory {name} - expected to find a file {name / MODEL_FILE}")
        with open(path / MODEL_FILE, "rb") as f:
            xml_data = f.read()
        model_data = unwrap_model_xml(model_type=MODEL_TYPE, xml_data=xml_data)
        model = Plugins().load(model_data)
        return SynthCityModel(model)
    def __repr__(self):
        return f"SynthCityModel({repr(self.model)})"
