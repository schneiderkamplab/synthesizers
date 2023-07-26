from datasets import Dataset

from .base import Pipeline

class TabularTrainingPipeline(Pipeline):
    def __call__(
        self,
        data: Dataset,
    ):
        input = self.adapter.convert_input(data['train'])
        model = self.adapter.train_model(input)
        return model

