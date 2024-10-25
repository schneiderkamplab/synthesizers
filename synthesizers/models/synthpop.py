from huggingface_hub import hf_hub_download
from pathlib import Path
from pickle import dumps, loads

from .base import Model
from ..utils.xml import unwrap_model_xml, wrap_model_xml

MODEL_FILE = "synthpop.xml"
MODEL_TYPE = "SynthPopModel"

class SynthPopModel(Model):
    def __init__(self, model):
        super(SynthPopModel, self).__init__(saveable=True)
        self.model = model
    def save_pretrained(self, name):
        path = Path(name)
        path.mkdir(parents=True, exist_ok=True)
        model_data = dumps(self.model)
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
        model = loads(model_data)
        return SynthPopModel(model)
    def __repr__(self):
        return f"SynthPopModel({repr(self.model)})"
