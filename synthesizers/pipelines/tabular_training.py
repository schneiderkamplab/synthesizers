from .base import Pipeline
from ..utils.loading import StateDict

class TabularTrainingPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
    ):
        state = StateDict.wrap(state)
        kwargs = dict(self.kwargs)
        kwargs.update(self.train_args)
        state.model = self.train_adapter.train_model(
            data=state.train,
            **kwargs,
        )
        return state

