from .base import Pipeline
from ..utils.formats import StateDict

class IdentityPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
    ):
        state = StateDict.wrap(state)
        return state
