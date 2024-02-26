from typing import List

from .base import Pipeline
from ..utils.formats import StateDict

class TabularEvaluationPipeline(Pipeline):

    def _call(
        self,
        state: StateDict,
    ) -> List[StateDict]:
        kwargs = dict(self.kwargs)
        kwargs.update(self.eval_args)
        state.eval = self.eval_adapter.evaluate_generated(
            original_data=state.train,
            generated_data=state.synth,
            hold_out=state.test,
            **kwargs,
        )
        return [state]
