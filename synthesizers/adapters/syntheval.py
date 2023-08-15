from pandas import DataFrame
from syntheval import SynthEval

from .base import Adapter

class SynthEvalAdapter(Adapter):
    def __init__(self, input_formats=(DataFrame,), config: str = 'full', unique_thresh: int = 10):
        super(SynthEvalAdapter, self).__init__(input_formats)
        self.config = config
        self.unique_thresh = unique_thresh
    def train_model(self, data):
        raise NotImplementedError("training not implemented for SynthEval")
    def generate_data(self, count, model):
        raise NotImplementedError("generation not implemented for SynthEval")
    def evaluate_generated(self, original_data, generated_data, config: str = None, hold_out = None, cat_cols: str = None, unique_thresh: int = 10, target_col: str = None):
        original_data = self.ensure_input_format(original_data)
        generated_data = self.ensure_input_format(generated_data)
        evaluator = SynthEval(real=original_data, hold_out=hold_out, cat_cols=cat_cols, unique_thresh=unique_thresh)
        config = self.config if config is None else config
        if config == 'full':
            evaluator.full_eval(synthetic_data=generated_data, target_col=target_col)
        elif config == 'fast':
            evaluator.fast_eval(synthetic_dataframe=generated_data, target=target_col)
        else:
            raise ValueError(f"unknown config {config} -- use either 'full' or 'fast'")
        return evaluator.res_dict
