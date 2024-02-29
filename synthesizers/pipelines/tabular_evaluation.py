from typing import List

from .base import Pipeline
from ..utils.formats import StateDict

class TabularEvaluationPipeline(Pipeline):

    def _call(
        self,
        state_dict: StateDict,
    ) -> List[StateDict]:
        kwargs = dict(self.kwargs)
        kwargs.update(self.eval_args)
        state_dict.eval = self.eval_adapter.evaluate_generated(
            original_data=state_dict.train,
            generated_data=state_dict.synth,
            hold_out=state_dict.test,
            **kwargs,
        )
        return [state_dict]
