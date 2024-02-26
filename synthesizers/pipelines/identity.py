from typing import List

from .base import Pipeline
from ..utils.formats import StateDict

class IdentityPipeline(Pipeline):

    def _call(
        self,
        state_dict: StateDict,
    ) -> List[StateDict]:
        if self.output_format is not None:
            if state_dict.train is not None:
                state_dict.train = self.ensure_output_format(state_dict.train, output_format=self.output_format)
            if state_dict.test is not None:
                state_dict.test = self.ensure_output_format(state_dict.test, output_format=self.output_format)
            if state_dict.synth is not None:
                state_dict.synth = self.ensure_output_format(state_dict.synth, output_format=self.output_format)
            if state_dict.eval is not None:
                state_dict.eval = self.ensure_output_format(state_dict.eval, output_format=self.output_format)
        return [state_dict]
