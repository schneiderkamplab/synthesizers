from .base import Pipeline
from ..utils.formats import StateDict

class IdentityPipeline(Pipeline):

    def _call(
        self,
        state: StateDict,
    ):
        if self.output_format is not None:
            if self.train is not None:
                self.train = self.ensure_output_format(self.train, output_format=self.output_format)
            if self.test is not None:
                self.test = self.ensure_output_format(self.test, output_format=self.output_format)
            if self.synth is not None:
                self.synth = self.ensure_output_format(self.synth, output_format=self.output_format)
            if self.eval is not None:
                self.eval = self.ensure_output_format(self.eval, output_format=self.output_format)
        return state
