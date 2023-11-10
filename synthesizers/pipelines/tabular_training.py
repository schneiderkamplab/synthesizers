from .base import Pipeline
from ..utils import loader

class TabularTrainingPipeline(Pipeline):
    def __call__(
        self,
        data: object,
    ):
        if type(data) == str:
            data = loader(data)
        model = self.adapter.train_model(
            data=data,
            **self.kwargs,
        )
        return model

