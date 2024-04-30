from typing import List, Optional, Union

from .base import Pipeline
from ..utils.formats import State, StateDict

class TabularSynthesisPipeline(Pipeline):

    def _call(
        self,
        state_dict: StateDict,
        gen_count: Optional[int] = None,
        split_size: Optional[Union[int, float]] = None,
        train_plugin: Optional[str] = None,
        do_eval: bool = True,
    ) -> List[StateDict]:
        from . import pipeline
        gen_args = dict(self.gen_args)
        if gen_count is None:
            if gen_args.get("count", None) is None:
                gen_args["count"] = 1 if state_dict.train is None else len(state_dict.train)
        else:
            gen_args["count"] = gen_count
        split_args = dict(self.split_args)
        if split_size is None:
            if split_args.get("size", None) is None:
                if state_dict.test is None:
                    split_args["size"] = 0.8
        else:
            split_args["size"] = split_size
        train_args = dict(self.train_args)
        if train_plugin is not None:
            train_args["plugin"] = train_plugin
        state = State(state_dicts=[state_dict])
        if split_args.get("size", None) is not None:
            split_pipe = pipeline("split", **split_args)
            state = split_pipe(state)
        train_pipe = pipeline("train", train_adapter=self.train_adapter, **train_args)
        state = train_pipe(state)
        gen_pipe = pipeline("generate", **gen_args)
        state = gen_pipe(state)
        if do_eval and self.kwargs.get("do_eval", True):
            eval_pipe = pipeline("evaluate", **self.eval_args)
            state = eval_pipe(state)
        state_dicts = state.state_dicts
        for state_dict in state_dicts:
            output_format = type(state_dict.train) if self.output_format is "auto" else self.output_format
            if output_format is not None:
                state_dict.synth = self.ensure_output_format(state_dict.synth, output_format=output_format)
        return state_dicts

class TabularSynthesisDPPipeline(TabularSynthesisPipeline):
    pass
