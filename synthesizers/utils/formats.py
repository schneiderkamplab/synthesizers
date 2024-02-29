from collections.abc import Iterable
from datasets import Dataset, DatasetDict, load_dataset
from numpy import ndarray, array
from os import PathLike
from os.path import isdir, isfile
from pandas import DataFrame, read_excel
from pathlib import Path
import pickle
from synthcity.plugins.core.dataloader import GenericDataLoader
from typing import Any, List, Union

from ..models import AutoModel, Model

SUPPORTED_FORMATS: Any = [
    list,
    ndarray,
    DataFrame,
    Dataset,
    GenericDataLoader,
]

STATE_DICT_FIELDS: str = [
    "train",
    "test",
    "synth",
    "model",
    "eval",
]

def ensure_format(
    data: Any,
    target_formats: List[Any],
    **kwargs,
) -> Any:
    assert all(target_format in SUPPORTED_FORMATS for target_format in target_formats)
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
    def __init__(
        self,
        train=None,
        test=None,
        synth=None,
        model=None,
        eval=None,
    ) -> None:
        assert model is None or isinstance(model, Model)
        self.train = train
        self.test = test
        self.synth = synth
        self.model = model
        self.eval = eval
    def wrap(data: Any):
        data_format = type(data)
        if data_format == StateDict:
            return data.clone()
        if data_format in SUPPORTED_FORMATS:
            return StateDict(train=data)
        if data_format in (DatasetDict, dict):
            return StateDict(
                train=data.get("train", None),
                test=data.get("test", None),
                synth=data.get("synth", None),
                model=data.get("model", None),
                eval=data.get("eval", None),
            )
        if isinstance(data, str) or isinstance(data, PathLike):
            return loader(data)
        raise ValueError(f"cannot wrap {data_format}")
    def clone(self):
        return StateDict(train=self.train, test=self.test, synth=self.synth, model=self.model, eval=self.eval)
    def __repr__(self):
        return repr(self.__dict__)
    def __str__(self):
        return str(self.__dict__)
    def save(
        self,
        name: Union[str, Path],
        output_format: Any = None,
        key: str = None,
    ) -> None:
        assert isinstance(name, str) or isinstance(name, PathLike)
        assert output_format is None or output_format in SUPPORTED_FORMATS
        assert key is None or key in STATE_DICT_FIELDS
        if key is not None:
            saver(self.__dict__[key], name, output_format=output_format)
        else:
            path = Path(name)
            path.mkdir(parents=True, exist_ok=True)
            for key in STATE_DICT_FIELDS:
                value = self.__dict__[key]
                if value is not None:
                    if key == "model":
                        value.save_pretrained(name)
                    else:
                        saver(value, path / f"{key}.pickle", output_format=output_format)
    def load(name: Union[str, Path]):
        assert isinstance(name, str) or isinstance(name, PathLike)
        path = Path(name)
        if not path.is_dir():
            raise RuntimeError(f"invalid state directory {name}")
        kwargs = {}
        for key in STATE_DICT_FIELDS:
            if (path / f"{key}.pickle").is_file():
                kwargs[key] = loader(path / f"{key}.pickle").train
        try:
            kwargs["model"] = AutoModel.from_pretrained(name)
        except:
            pass
        state_dict = StateDict(**kwargs)
        return state_dict

class State():
    def __init__(
        self,
        state_dicts: List[StateDict],
    ) -> None:
        assert state_dicts
        self.state_dicts = state_dicts
    def wrap(data: Any):
        if isinstance(data, State):
            return data.clone()
        if isinstance(data, tuple):
            state_dicts = [StateDict.wrap(d) for d in data]
        elif (isinstance(data, str) or isinstance(data, PathLike)) and isdir(data) and isdir(Path(data) / "0"):
            i = 0
            state_dicts = []
            while isdir(Path(data) / str(i)):
                state_dicts.append(StateDict.load(Path(data) / str(i)))
                i += 1
        else:
            state_dicts = [StateDict.wrap(data)]
        return State(state_dicts=state_dicts)
    def clone(self):
        state_dicts = [state_dict.clone() for state_dict in self.state_dicts]
        return State(state_dicts=state_dicts)
    def __repr__(self):
        return repr(self.state_dicts)
    def __str__(self):
        return str(self.state_dicts)
    def __len__(self):
        return len(self.state_dicts)
    def __getitem__(self, index):
        return self.state_dicts[index]
    def __setitem__(self, index, value):
        self.state_dicts[index] = value
    def Save(
        self,
        name: Union[str, Path],
        output_format: Any = None,
        key: str = None,
        index: int =None,
    ) -> None:
        assert isinstance(name, str) or isinstance(name, PathLike)
        assert output_format is None or output_format in SUPPORTED_FORMATS
        assert key is None or key in STATE_DICT_FIELDS
        assert index is None or (isinstance(index, int) and 0 <= index < len(self.state_dicts))
        if index is not None:
            self.state_dicts[index].save(name, output_format=output_format, key=key)
        elif len(self.state_dicts) == 1:
            self.state_dicts[0].save(name, output_format=output_format, key=key)
        elif key is not None:
            file_ext = str(name).split(".")[-1]
            file_name = str(name).split(f'.{file_ext}')[0]
            for i, state_dict in enumerate(self.state_dicts):
                state_dict.save(f'{file_name}_{i}.{file_ext}', output_format=output_format, key=key)
        else:
            for i, state_dict in enumerate(self.state_dicts):
                state_dict.save(f'{name}/{i}', output_format=output_format)
        return self
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

def saver(
    data: Any,
    name: Union[str, Path],
    output_format = None,
    key: str = None,
) -> None:
    assert isinstance(name, str) or isinstance(name, PathLike)
    assert output_format is None or output_format in SUPPORTED_FORMATS
    assert key is None or key in STATE_DICT_FIELDS
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
        elif str(name).endswith(".tsv"):
            data = ensure_format(data, target_formats=[DataFrame])
            data.to_csv(name, sep="\t")
        elif str(name).endswith(".pickle"):
            if output_format is not None:
                data = ensure_format(data, output_format)
            with open(name, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"unknown format for {name}")

def loader(name: Union[str, Path]) -> StateDict:
    assert isinstance(name, str) or isinstance(name, PathLike)
    if isfile(name):
        extension = str(name).split(".")[1].lower()
        if extension == "pickle":
            with open(name, "rb") as f:
                return StateDict.wrap(pickle.load(f))
        if extension == "xlsx":
            return StateDict.wrap(read_excel(name))
        if extension in ("json", "jsonl"):
            return StateDict.wrap(load_dataset("json", data_files=name))
        if extension in ("csv", "tsv"):
            return StateDict.wrap(load_dataset("csv", data_files=name))
        raise ValueError(f"unknown format for {name}")
    try:
        return StateDict.wrap(load_dataset(name))
    except:
        pass
    if isdir(name):
        return StateDict.load(name)
    raise ValueError(f"cannot determine how to load {name}")