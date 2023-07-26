from datasets import Dataset
from pandas import DataFrame
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins

from .base import Adapter

class SynthCityAdapter(Adapter):
    def __init__(self, plugin = 'adsgan'):
        self.plugin = plugin
    def convert_input(self, data, **kwargs):
        df = data.to_pandas()
        loader = GenericDataLoader(
            df,
            **kwargs
        )
        return loader
    def train_model(self, data):
        model = Plugins().get(self.plugin)
        model.fit(data)
        return model
    def generate_data(self, model, count):
        result = model.generate(count=count)
        df = result.dataframe()
        return df
    def convert_output(self, data):
        ds = Dataset.from_pandas(data)
        ds = ds.remove_columns(['__index_level_0__'])
        return ds