from .base import Pipeline

class TabularEvaluationPipeline(Pipeline):
    def __call__(
        self,
        original_data: object,
        generated_data: object,
    ):
        return self.adapter.evaluate_generated(
            original_data=original_data,
            generated_data=generated_data,
            **self.kwargs,
        )
