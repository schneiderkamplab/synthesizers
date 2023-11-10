from .base import Pipeline
from ..utils.loading import StateDict

class TabularTrainingPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
    ):
        state = StateDict.wrap(state)
        state.model = self.train_adapter.train_model(
            data=state.train,
            **self.kwargs,
        )
        return state

