import io
from contextlib import nullcontext, redirect_stdout
import warnings

from pandas import DataFrame
from syntheval import SynthEval

from .base import Adapter

__all__ = ["SynthEvalAdapter"]

def _syntheval_warning_context(suppress_warnings: bool):
    if not suppress_warnings:
        return nullcontext()

    class _Context:
        def __enter__(self):
            self._warnings = warnings.catch_warnings()
            self._warnings.__enter__()
            warnings.filterwarnings(
                "ignore",
                message=r"The number of unique classes is greater than 50% of the number of samples.*",
                category=UserWarning,
                module=r"sklearn\.metrics\.cluster\._supervised",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Clustering metrics expects discrete values.*",
                category=UserWarning,
                module=r"sklearn\.metrics\.cluster\._supervised",
            )
            self._stdout = redirect_stdout(io.StringIO())
            self._stdout.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            self._stdout.__exit__(exc_type, exc_value, traceback)
            self._warnings.__exit__(exc_type, exc_value, traceback)

    return _Context()

class SynthEvalAdapter(Adapter):
    def __init__(self, input_formats=(DataFrame,), presets_file: str = 'full_eval', unique_threshold: int = 10):
        super(SynthEvalAdapter, self).__init__(input_formats)
        self.config = presets_file
        self.unique_thresh = unique_threshold
    def train_model(self, data):
        raise NotImplementedError("training not implemented for SynthEval")
    def generate_data(self, count, model):
        raise NotImplementedError("generation not implemented for SynthEval")
    def evaluate_generated(self, original_data, generated_data, config: str = None, hold_out = None, cat_cols: str = None, unique_threshold: int = 10, target_col: str = None, suppress_warnings: bool = True):
        original_data = self.ensure_input_format(original_data)
        generated_data = self.ensure_input_format(generated_data)
        evaluator = SynthEval(real_dataframe=original_data, holdout_dataframe=hold_out, cat_cols=cat_cols, unique_threshold=unique_threshold, verbose=False)
        config = self.config if config is None else config
        with _syntheval_warning_context(suppress_warnings):
            results = evaluator.evaluate(synthetic_dataframe=generated_data, analysis_target_var=target_col, presets_file=config)
        return results
