from datasets import DatasetDict

from .base import Pipeline

class TabularGenerationPipeline(Pipeline):
    def __call__(
        self,
        model: object,
        count: int = 1,
    ):
        output = self.adapter.generate_data(model, count)
        ds = self.adapter.convert_output(output)
        dd = DatasetDict()
        dd['generated'] = ds
        return dd
