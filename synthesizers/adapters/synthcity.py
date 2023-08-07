from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins

from .base import Adapter
from ..models.synthcity import SynthCityModel

class SynthCityAdapter(Adapter):
    def __init__(self, plugin = 'adsgan', evaluator_class = AlphaPrecision, input_formats=(GenericDataLoader,)):
        super(SynthCityAdapter, self).__init__(input_formats)
        self.plugin = plugin
        self.evaluator_class = evaluator_class
    def train_model(self, data, plugin=None):
        data = self.ensure_input_format(data)
        model = Plugins().get(self.plugin if plugin is None else plugin)
        model.fit(data)
        return SynthCityModel(model)
    def generate_data(self, count, model):
        result = model.model.generate(count=count)
        return result
    def evaluate_generated(self, original_data, generated_data, evaluator_class=None):
        original_data = self.ensure_input_format(original_data)
        generated_data = self.ensure_input_format(generated_data)
        evaluator_class = self.evaluator_class if evaluator_class is None else evaluator_class
        evaluator = evaluator_class()
        return evaluator.evaluate(original_data, generated_data)
