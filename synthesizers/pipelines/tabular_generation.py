from .base import Pipeline
from ..models.base import Model

class TabularGenerationPipeline(Pipeline):
    def __call__(
        self,
        count: int = 1,
    ):
        output = self.adapter.generate_data(
            count=count,
            **self.kwargs,
        )
        if self.output_format is not None:
            output = self.ensure_output_format(output)
        return output
