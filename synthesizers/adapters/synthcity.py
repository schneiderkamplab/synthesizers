from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins

from .base import Adapter

class SynthCityAdapter(Adapter):
    def __init__(self, plugin = 'adsgan', evaluator_class = AlphaPrecision, input_formats=(GenericDataLoader,)):
        super(SynthCityAdapter, self).__init__(input_formats)
        self.plugin = plugin
        self.evaluator_class = evaluator_class
    def train_model(self, data, **kwargs):
        data = self.ensure_input_format(data, **kwargs)
        model = Plugins().get(self.plugin)
        model.fit(data)
        return model
    def generate_data(self, model, count, **_):
        result = model.generate(count=count)
        return result
    def evaluate_generated(self, orig_data, data, **kwargs):
        orig_data = self.ensure_input_format(orig_data, **kwargs)
        data = self.ensure_input_format(data, **kwargs)
        evaluator = self.evaluator_class()
        return evaluator.evaluate(orig_data, data)
