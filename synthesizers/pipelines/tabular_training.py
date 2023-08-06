from datasets import Dataset

from .base import Pipeline

class TabularTrainingPipeline(Pipeline):
    def __call__(
        self,
        data: object,
    ):
        model = self.adapter.train_model(data, **self.kwargs)
        return model

