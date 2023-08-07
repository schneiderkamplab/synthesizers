from huggingface_hub import snapshot_download
from pathlib import Path

from .base import Model
from .synthcity import SynthCityModel
from .synthpop import SynthPopModel

FILE_TO_CLASS = {
    "synthcity.pickle": SynthCityModel,
    "synthpop.pickle": SynthPopModel,
}

MODEL_TO_ADAPTER = {
    SynthCityModel: "SynthCityAdapter",
    SynthPopModel: "SynthPopAdapter",
}

class AutoModel(Model):
    def __init__(self):
        super(AutoModel, self).__init__(saveable=False)
    def from_pretrained(name):
        path = Path(name)
        if not path.exists():
            snapshot_download(repo_id=name)
        if not path.is_dir():
            raise RuntimeError(f"invalid model directory {name} - expected to find a directory {name}")
        for file_name, model_class in FILE_TO_CLASS.items():
            if (path / file_name).is_file():
                return model_class.from_pretrained(name)
        raise RuntimeError(f"could not determine model type for {name} - expected on of: {', '.join(FILE_TO_CLASS.keys())}")
