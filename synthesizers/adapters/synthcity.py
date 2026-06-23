from pathlib import Path

from .base import Adapter
from ..models import SynthCityModel
from ..utils import (
    GenericDataLoaderFormat,
    TimeSeriesDataLoaderFormat,
    quiet_synthcity_imports,
)

__all__ = ["SynthCityAdapter", "SynthCityMetricsAdapter"]

_PLUGIN_CATEGORIES = {
    "adsgan": ["privacy"],
    "aim": ["privacy"],
    "arf": ["generic"],
    "bayesian_network": ["generic"],
    "ctgan": ["generic"],
    "ddpm": ["generic"],
    "decaf": ["privacy"],
    "dpgan": ["privacy"],
    "dummy_sampler": ["generic"],
    "fflows": ["time_series"],
    "goggle": ["generic"],
    "great": ["generic"],
    "image_adsgan": ["images"],
    "image_cgan": ["images"],
    "marginal_distributions": ["generic"],
    "nflow": ["generic"],
    "pategan": ["privacy"],
    "privbayes": ["privacy"],
    "radialgan": ["domain_adaptation"],
    "rtvae": ["generic"],
    "survae": ["survival_analysis"],
    "survival_ctgan": ["survival_analysis"],
    "survival_gan": ["survival_analysis"],
    "survival_nflow": ["survival_analysis"],
    "timegan": ["time_series"],
    "timevae": ["time_series"],
    "tvae": ["generic"],
    "uniform_sampler": ["generic"],
}

def _load_plugin(plugin, **kwargs):
    with quiet_synthcity_imports():
        from synthcity.plugins import Plugins
        categories = _PLUGIN_CATEGORIES.get(plugin, None)
        plugins = Plugins(categories=categories) if categories is not None else Plugins()
        return plugins.get(plugin, **kwargs)

class SynthCityAdapter(Adapter):
    def __init__(self, plugin = 'adsgan', evaluator_class = None, input_formats=(GenericDataLoaderFormat, TimeSeriesDataLoaderFormat)):
        super(SynthCityAdapter, self).__init__(input_formats)
        self.plugin = plugin
        self.evaluator_class = evaluator_class
    def train_model(self, data, plugin=None, **kwargs):
        data = self.ensure_input_format(data)
        plugin = self.plugin if plugin is None else plugin
        model = _load_plugin(plugin, **kwargs)
        model.fit(data)
        return SynthCityModel(model)
    def generate_data(self, count, model):
        result = model.model.generate(count=count)
        return result
    def evaluate_generated(self, original_data, generated_data, holdout, evaluator_class=None):
        original_data = self.ensure_input_format(original_data)
        generated_data = self.ensure_input_format(generated_data)
        if self.evaluator_class is None and evaluator_class is None:
            with quiet_synthcity_imports():
                from synthcity.metrics.eval_statistical import AlphaPrecision
            evaluator_class = AlphaPrecision
        else:
            evaluator_class = self.evaluator_class if evaluator_class is None else evaluator_class
        evaluator = evaluator_class()
        return evaluator.evaluate(original_data, generated_data)

class SynthCityMetricsAdapter(Adapter):
    def __init__(self, input_formats=(GenericDataLoaderFormat, TimeSeriesDataLoaderFormat)):
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
        with quiet_synthcity_imports():
            from synthcity.metrics.eval import Metrics
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
