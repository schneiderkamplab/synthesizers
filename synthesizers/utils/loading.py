from datasets import Dataset, DatasetDict, load_dataset
from numpy import ndarray
from os.path import isfile
from pandas import DataFrame, read_excel
from synthcity.plugins.core.dataloader import GenericDataLoader

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

def loader(data):
    if isfile(data):
        extension = data.split(".")[1].lower()
        if extension == "xlsx":
            return StateDict.wrap(read_excel(data))
        if extension in ("json", "jsonl"):
            return StateDict.wrap(load_dataset("json", data_files=data))
        if extension in ("csv", "tsv"):
            return StateDict.wrap(load_dataset("csv", data_files=data))
        raise ValueError(f"unknown format for {data}")
    return StateDict.wrap(load_dataset(data))