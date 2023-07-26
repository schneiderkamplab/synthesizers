from datasets import Dataset, DatasetDict

from .base import Pipeline

class TabularSynthesisPipeline(Pipeline):
    def __call__(
        self,
        data: Dataset,
        count: int = None,
        include_real: bool = True,
    ):
        if count is None:
            count = len(data['train'])
        input = self.adapter.convert_input(data['train'])
        model = self.adapter.train_model(input)
        output = self.adapter.generate_data(model, count)
        ds = self.adapter.convert_output(output)
        dd = DatasetDict(data) if include_real else DatasetDict()
        dd['generated'] = ds
        return dd

class TabularSynthesisDPPipeline(TabularSynthesisPipeline):
    pass