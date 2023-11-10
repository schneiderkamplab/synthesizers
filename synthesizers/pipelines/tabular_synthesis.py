from typing import Optional

from .base import Pipeline
from ..utils.loading import StateDict

class TabularSynthesisPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
        count: Optional[int] = None,
        do_eval: bool = True,
    ):
        state = StateDict.wrap(state)
        if count is None:
            # assumption that all formats implement __len__
            count = len(state.train)
        state.model = self.train_adapter.train_model(
            data=state.train,
            **self.train_args,
        )
        state.synth = self.train_adapter.generate_data(
            count=count,
            model=state.model,
            **self.gen_args,
        )
        if do_eval and self.kwargs.get("do_eval", True):
            state.eval = self.eval_adapter.evaluate_generated(
                original_data=state.train,
                generated_data=state.synth,
                hold_out=state.test,
                **self.eval_args,
            )
        output_format = type(state.train) if self.output_format is "auto" else self.output_format
        if output_format is not None:
            self.synth = self.ensure_output_format(self.synth, output_format=output_format)
        return state

class TabularSynthesisDPPipeline(TabularSynthesisPipeline):
    pass