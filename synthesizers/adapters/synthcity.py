from pathlib import Path

from synthcity.metrics.eval import Metrics
from synthcity.metrics.eval_statistical import AlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader, TimeSeriesDataLoader
from synthcity.plugins import Plugins

from .base import Adapter
from ..models import SynthCityModel

__all__ = ["SynthCityAdapter", "SynthCityMetricsAdapter"]

class SynthCityAdapter(Adapter):
    def __init__(self, plugin = 'adsgan', evaluator_class = AlphaPrecision, input_formats=(GenericDataLoader, TimeSeriesDataLoader)):
        super(SynthCityAdapter, self).__init__(input_formats)
        self.plugin = plugin
        self.evaluator_class = evaluator_class
    def train_model(self, data, plugin=None, **kwargs):
        data = self.ensure_input_format(data)
        model = Plugins().get(self.plugin if plugin is None else plugin, **kwargs)
        model.fit(data)
        return SynthCityModel(model)
    def generate_data(self, count, model):
        result = model.model.generate(count=count)
        return result
    def evaluate_generated(self, original_data, generated_data, holdout, evaluator_class=None):
        original_data = self.ensure_input_format(original_data)
        generated_data = self.ensure_input_format(generated_data)
        evaluator_class = self.evaluator_class if evaluator_class is None else evaluator_class
        evaluator = evaluator_class()
        return evaluator.evaluate(original_data, generated_data)

class SynthCityMetricsAdapter(Adapter):
    def __init__(self, input_formats=(GenericDataLoader, TimeSeriesDataLoader)):
        super(SynthCityMetricsAdapter, self).__init__(input_formats)
    def train_model(self, data):
        raise NotImplementedError("training not implemented for SynthCityMetricsAdapter")
    def generate_data(self, count, model):
        raise NotImplementedError("generation not implemented for SynthCityMetricsAdapter")
    def evaluate_generated(
        self,
        original_data,
        generated_data,
        hold_out=None,
        metrics=None,
        task_type=None,
        workspace=Path("workspace"),
        **kwargs,
    ):
        original_data = self.ensure_input_format(original_data)
        generated_data = self.ensure_input_format(generated_data)
        if task_type is None:
            task_type = "time_series" if original_data.type() == "time_series" else "classification"
        return Metrics.evaluate(
            X_gt=original_data,
            X_syn=generated_data,
            metrics=metrics,
            task_type=task_type,
            workspace=workspace,
            **kwargs,
        )
