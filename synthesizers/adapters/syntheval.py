from pandas import DataFrame
from syntheval import SynthEval

from .base import Adapter

class SynthEvalAdapter(Adapter):
    def __init__(self, input_formats=(DataFrame,), target_col: str = None):
        super(SynthEvalAdapter, self).__init__(input_formats)
        self.target_col = target_col
    def train_model(self, data):
        raise NotImplementedError("training not implemented for SynthEval")
    def generate_data(self, count, model):
        raise NotImplementedError("generation not implemented for SynthEval")
    def evaluate_generated(self, original_data, generated_data, target_col: str = None):
        original_data = self.ensure_input_format(original_data)
        generated_data = self.ensure_input_format(generated_data)
        evaluator = SynthEval(real=original_data)
        target_col = self.target_col if target_col is None else target_col
        evaluator.full_eval(synthetic_data=generated_data, target_col=target_col)
        return evaluator.res_dict
