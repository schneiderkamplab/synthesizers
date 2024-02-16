from datasets import Dataset, DatasetDict, load_dataset
from numpy import ndarray, array
from os.path import isdir, isfile
from pandas import DataFrame, read_excel
from pathlib import Path
import pickle
from synthcity.plugins.core.dataloader import GenericDataLoader

from ..models import AutoModel

SUPPORTED_FORMATS = [
    list,
    ndarray,
    DataFrame,
    Dataset,
    GenericDataLoader,
]

def ensure_format(data, target_formats, **kwargs):
    source_format = type(data)
    if source_format == str:
        data = loader(data)
        source_format = type(data)
    if source_format not in SUPPORTED_FORMATS:
        raise ValueError(f"unknown source format {source_format}")
    for target_format in target_formats:
        if target_format not in SUPPORTED_FORMATS:
            raise ValueError(f"unknwown target format {target_format} in {target_formats}")
    if source_format in target_formats:
        return data
    for target_format in target_formats:
        if target_format == Dataset:
            if source_format in (list, ndarray):
                data, source_format = DataFrame(data), DataFrame
            if source_format == GenericDataLoader:
                data, source_format = data.dataframe(), DataFrame
            if source_format == DataFrame:
                ds = Dataset.from_pandas(data)
                ds = ds.remove_columns([feature for feature in ds.features if feature.startswith("__") and feature.endswith("__")])
                return ds
            raise ValueError(f"cannot convert {source_format} to {target_format}")
        if target_format == DataFrame:
            if source_format in (list, ndarray):
                return DataFrame(data)
            if source_format == Dataset:
                return data.to_pandas()
            if source_format == GenericDataLoader:
                return data.dataframe()
            raise ValueError(f"cannot convert {source_format} to {target_format}")
        if target_format == GenericDataLoader:
            if source_format == Dataset:
                data, source_format = data.to_pandas(), DataFrame
            if source_format in (list, ndarray, DataFrame):
                return GenericDataLoader(data, **kwargs)
            raise ValueError(f"cannot convert {source_format} to {target_format}")
        if target_format == list:
            if source_format == Dataset:
                data, source_format = data.to_pandas(), DataFrame
            if source_format == GenericDataLoader:
                data, source_format = data.dataframe(), DataFrame
            if source_format == DataFrame:
                data, source_format = data.to_numpy(), ndarray
            if source_format == ndarray:
                return data.tolist()
            raise ValueError(f"cannot convert {source_format} to {target_format}")
        if target_format == ndarray:
            if source_format == Dataset:
                data, source_format = data.to_pandas(), DataFrame
            if source_format == GenericDataLoader:
                data, source_format = data.dataframe(), DataFrame
            if source_format == DataFrame:
                return data.to_numpy()
            if source_format == list:
                return array(data)
            raise ValueError(f"cannot convert {source_format} to {target_format}")
        raise ValueError(f"cannot convert {source_format} to any of {target_formats}")

class StateDict():
    def __init__(self, train=None, test=None, synth=None, model=None, eval=None):
        self.train = train
        self.test = test
        self.synth = synth
        self.model = model
        self.eval = eval
    def wrap(data):
        data_format = type(data)
        if data_format == StateDict:
            return data.clone()
        if data_format in (list, ndarray, DataFrame, Dataset, GenericDataLoader):
            return StateDict(train=data)
        if data_format in (DatasetDict, dict):
            return StateDict(
                train=data.get("train", None),
                test=data.get("test", None),
                synth=data.get("synth", None),
                model=data.get("model", None),
                eval=data.get("eval", None),
            )
        if data_format == str:
            return loader(data)
    def clone(self):
        return StateDict(train=self.train, test=self.test, synth=self.synth, model=self.model, eval=self.eval)
    def __repr__(self):
        return repr(self.__dict__)
    def __str__(self):
        return str(self.__dict__)
    def Save(self, name, output_format=None, key=None):
        if key is not None:
            saver(self.__dict__[key], name, output_format=output_format)
        else:
            path = Path(name)
            path.mkdir(parents=True, exist_ok=True)
            for key, value in self.__dict__.items():
                if value is not None:
                    if key == "model":
                        value.save_pretrained(name)
                    else:
                        saver(value, path / f"{key}.pickle", output_format=output_format)
    def Synthesize(self, **kwargs):
        from ..pipelines import pipeline
        return pipeline("synthesize", **kwargs)(self)
    def Split(self, **kwargs):
        from ..pipelines import pipeline
        return pipeline("split", **kwargs)(self)
    def Evaluate(self, **kwargs):
        from ..pipelines import pipeline
        return pipeline("evaluate", **kwargs)(self)
    def Train(self, **kwargs):
        from ..pipelines import pipeline
        return pipeline("train", **kwargs)(self)
    def Generate(self, **kwargs):
        from ..pipelines import pipeline
        return pipeline("generate", **kwargs)(self)
    def load(name):
        path = Path(name)
        if not path.is_dir():
            raise RuntimeError(f"invalid statel directory {name}")
        kwargs = {}
        for key in ("train", "test", "synth", "eval"):
            if (path / f"{key}.pickle").is_file():
                kwargs[key] = loader(path / f"{key}.pickle")
        try:
            kwargs["model"] = AutoModel.from_pretrained(name)
        except:
            pass
        state = StateDict(**kwargs)
        return state

def saver(data, name, output_format=None, key=None):
    if isinstance(data, StateDict):
        data.save(name, output_format=output_format, key=key)
    else:
        if str(name).endswith(".xlsx"):
            data = ensure_format(data, target_formats=[DataFrame])
            data.to_excel(name)
        elif str(name).endswith(".json"):
            data = ensure_format(data, target_formats=[Dataset])
            data.to_json(name, orient="records", lines=False)
        elif str(name).endswith(".jsonl"):
            data = ensure_format(data, target_formats=[Dataset])
            data.to_json(name, orient="records", lines=True)
        elif str(name).endswith(".csv"):
            data = ensure_format(data, target_formats=[DataFrame])
            data.to_csv(name)
        else:
            if output_format is not None:
                data = ensure_format(data, output_format)
            with open(name, "wb") as f:
                pickle.dump(data, f)

def loader(data):
    if isfile(data):
        extension = str(data).split(".")[1].lower()
        if extension == "pickle":
            with open(data, "rb") as f:
                return StateDict.wrap(pickle.load(f))
        if extension == "xlsx":
            return StateDict.wrap(read_excel(data))
        if extension in ("json", "jsonl"):
            return StateDict.wrap(load_dataset("json", data_files=data))
        if extension in ("csv", "tsv"):
            return StateDict.wrap(load_dataset("csv", data_files=data))
        raise ValueError(f"unknown format for {data}")
    try:
        return StateDict.wrap(load_dataset(data))
    except:
        pass
    if isdir(data):
        return StateDict.load(data)
    raise ValueError(f"cannot determine how to load {data}")