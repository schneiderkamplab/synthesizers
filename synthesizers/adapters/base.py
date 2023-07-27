from datasets import Dataset
from pandas import DataFrame

class Adapter:
    def __init__(self, format):
        self.format = format
    def ensure_format(self, data, format):
        if format == 'datasets':
            if isinstance(data, Dataset):
                return data
            elif isinstance(data, DataFrame):
                ds = Dataset.from_pandas(data)
                ds = ds.remove_columns([feature for feature in ds.features if feature.startswith("__") and feature.endswith("__")])
                return ds
            else:
                raise ValueError(f"cannot convert {type(data)} to {format}")
        elif format == "pandas":
            if isinstance(data, DataFrame):
                return data
            elif isinstance(data, Dataset):
                return data.to_pandas()
            else:
                raise ValueError(f"cannot convert {type(data)} to {format}")
        else:
            raise ValueError(f"unknown format {format}")