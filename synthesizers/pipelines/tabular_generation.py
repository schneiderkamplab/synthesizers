from ..adapters import *
from .base import Pipeline
from ..models.auto import MODEL_TO_ADAPTER

class TabularGenerationPipeline(Pipeline):
    def __call__(
        self,
        count: int = 1,
    ):
        if self.adapter is None:
            self.adapter = eval(MODEL_TO_ADAPTER[self.kwargs["model"].__class__])()
        output = self.adapter.generate_data(
            count=count,
            **self.kwargs,
        )
        if self.output_format is not None:
            output = self.ensure_output_format(output)
        return output
