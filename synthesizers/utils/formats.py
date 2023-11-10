from datasets import Dataset
from numpy import ndarray, array
from pandas import DataFrame
from synthcity.plugins.core.dataloader import GenericDataLoader

from .loading import loader

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