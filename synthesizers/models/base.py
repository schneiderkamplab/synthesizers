from huggingface_hub import hf_hub_download
from pathlib import Path

from ..utils.xml import unwrap_model_xml, wrap_model_xml

class Model:
    def __init__(self, saveable):
        self.saveable = saveable
    def save_pretrained(self, name, part_size):
        path = Path(name)
        path.mkdir(parents=True, exist_ok=True)
        model_data = self.dump_model_data()
        xml_data = wrap_model_xml(model_type=self.MODEL_TYPE, model_data=model_data, part_size=part_size)
        if len(xml_data) == 1:
            with open(path / self.MODEL_FILE, "wb") as f:
                f.write(xml_data[0])
        else:
            for i, xml_datum in enumerate(xml_data, start=1):
                with open(path / (self.MODEL_FILE_PART.format(i)), "wb") as f:
                    f.write(xml_datum)
    @classmethod
    def from_pretrained(cls, name):
        path = Path(name)
        if not path.exists():
            hf_hub_download(repo_id=name, filename=cls.MODEL_FILE)
        if not path.is_dir() or not ((path / cls.MODEL_FILE).is_file() or (path / cls.MODEL_FILE_PART.format(1)).is_file()):
            raise RuntimeError(f"invalid synthcity model directory {name} - expected to find a file {name / cls.MODEL_FILE} or {name / cls.MODEL_FILE_PART.format(1)}")
        if (path / cls.MODEL_FILE).is_file():
            with open(path / cls.MODEL_FILE, "rb") as f:
                xml_data = [f.read()]
        else:
            xml_data = []
            i = 1
            while (path / cls.MODEL_FILE_PART.format(i)).is_file():
                with open(path / cls.MODEL_FILE_PART.format(i), "rb") as f:
                    xml_data.append(f.read())
                i += 1
        model_data = unwrap_model_xml(model_type=cls.MODEL_TYPE, xml_data=xml_data)
        return cls.load_model_data(model_data)
