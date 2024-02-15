from pandas import DataFrame
from syntheval import SynthEval

from .base import Adapter

class SynthEvalAdapter(Adapter):
    def __init__(self, input_formats=(DataFrame,), presets_file: str = 'full_eval', unique_threshold: int = 10):
        super(SynthEvalAdapter, self).__init__(input_formats)
        self.config = presets_file
        self.unique_thresh = unique_threshold
    def train_model(self, data):
        raise NotImplementedError("training not implemented for SynthEval")
    def generate_data(self, count, model):
        raise NotImplementedError("generation not implemented for SynthEval")
    def evaluate_generated(self, original_data, generated_data, config: str = None, hold_out = None, cat_cols: str = None, unique_threshold: int = 10, target_col: str = None):
        original_data = self.ensure_input_format(original_data)
        generated_data = self.ensure_input_format(generated_data)
        evaluator = SynthEval(real_dataframe=original_data, holdout_dataframe=hold_out, cat_cols=cat_cols, unique_threshold=unique_threshold, verbose=False)
        config = self.config if config is None else config
        results = evaluator.evaluate(synthetic_dataframe=generated_data, analysis_target_var=target_col, presets_file=config)
        return results
