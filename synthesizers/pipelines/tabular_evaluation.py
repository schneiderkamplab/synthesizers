from datasets import DatasetDict

from .base import Pipeline

class TabularEvaluationPipeline(Pipeline):
    def __call__(
        self,
        original_data: object,
        generated_data: object,
        count: int = 1,
    ):
        return self.adapter.evaluate_generated(original_data, generated_data)
