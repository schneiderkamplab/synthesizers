from datasets import Dataset
from pandas import DataFrame
from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins

from .base import Adapter

class SynthCityAdapter(Adapter):
    def __init__(self, plugin = 'adsgan', format='datasets'):
        super(SynthCityAdapter, self).__init__(format)
        self.plugin = plugin
    def convert_input(self, data, **kwargs):
        df = self.ensure_format(data, 'pandas')
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
        return self.ensure_format(data, self.format)
    def evaluate_generated(self, orig_data, data, **kwargs):
        print(orig_data, data)
        orig_df = self.ensure_format(orig_data, 'pandas')
        df = self.ensure_format(data, 'pandas')
        print(orig_df, df)
        orig_loader = GenericDataLoader(
            orig_df,
            **kwargs
        )
        loader = GenericDataLoader(
            df,
            **kwargs
        )
        evaluator = AlphaPrecision()
        return evaluator.evaluate(orig_loader, loader)
