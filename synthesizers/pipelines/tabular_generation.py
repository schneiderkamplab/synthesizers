from .base import Pipeline

class TabularGenerationPipeline(Pipeline):
    def __call__(
        self,
        model: object,
        count: int = 1,
    ):
        output = self.adapter.generate_data(model, count)
        if self.output_format is not None:
            output = self.ensure_output_format(output)
        return output
