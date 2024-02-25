from .base import Pipeline
from ..utils.formats import StateDict

class TabularTrainingPipeline(Pipeline):

    def _call(
        self,
        state: StateDict,
    ):
        kwargs = dict(self.kwargs)
        kwargs.update(self.train_args)
        state.model = self.train_adapter.train_model(
            data=state.train,
            **kwargs,
        )
        return state

