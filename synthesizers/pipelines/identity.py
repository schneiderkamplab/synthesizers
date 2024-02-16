from .base import Pipeline
from ..utils.formats import StateDict

class IdentityPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
    ):
        state = StateDict.wrap(state)
        if self.output_format is not None:
            self.synth = self.ensure_output_format(self.synth, output_format=self.output_format)
        if self.save_args.get("name", None) is not None:
            state.Save(**self.save_args)
        return state
