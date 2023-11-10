from .base import Pipeline
from ..utils.loading import StateDict

class TabularEvaluationPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
    ):
        state = StateDict.wrap(state)
        kwargs = dict(self.kwargs)
        kwargs.update(self.eval_args)
        state.eval = self.eval_adapter.evaluate_generated(
            original_data=state.train,
            generated_data=state.synth,
            hold_out=state.test,
            **kwargs,
        )
        return state
