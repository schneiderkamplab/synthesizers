from datasets import load_dataset
from os.path import isfile
from pandas import read_excel

def loader(data, split="train"):
    if isfile(data):
        extension = data.split(".")[1].lower()
        if extension == "xlsx":
            return read_excel(data)
        if extension in ("json", "jsonl"):
            return load_dataset("json", data_files=data, split=split)
        if extension in ("csv", "tsv"):
            return load_dataset("csv", data_files=data, split=split)
        raise ValueError(f"unknown format for {data}")
    return load_dataset(data, split=split)
