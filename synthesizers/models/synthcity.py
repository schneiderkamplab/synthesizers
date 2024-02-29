from huggingface_hub import hf_hub_download
from pathlib import Path
from synthcity.plugins import Plugins

from .base import Model

MODEL_FILE = "synthcity.pickle"

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
        buff = self.model.save()
        with open(path / MODEL_FILE, "wb") as f:
            f.write(buff)
    def from_pretrained(name):
        path = Path(name)
        if not path.exists():
            hf_hub_download(repo_id=name, filename=MODEL_FILE)
        if not path.is_dir() or not (path / MODEL_FILE).is_file():
            raise RuntimeError(f"invalid synthcity model directory {name} - expected to find a file {name / MODEL_FILE}")
        with open(path / MODEL_FILE, "rb") as f:
            buff = f.read()
        model = Plugins().load(buff)
        return SynthCityModel(model)
    def __repr__(self):
        return f"SynthCityModel({repr(self.model)})"
