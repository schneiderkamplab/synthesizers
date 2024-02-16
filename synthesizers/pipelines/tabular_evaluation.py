from .base import Pipeline
from ..utils.formats import StateDict

class TabularEvaluationPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
    ):
        print(type(state))
        print(type(state))
        state = StateDict.wrap(state)
        kwargs = dict(self.kwargs)
        kwargs.update(self.eval_args)
        state.eval = self.eval_adapter.evaluate_generated(
            original_data=state.train,
            generated_data=state.synth,
            hold_out=state.test,
            **kwargs,
        )
        if self.save_args.get("name", None) is not None:
            state.Save(**self.save_args)
        return state
