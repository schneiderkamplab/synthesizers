from itertools import chain
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import List, Optional, Union

from .base import Pipeline
from ..utils.formats import State, StateDict

class TabularSynthesisPipeline(Pipeline):

    def _call(
        self,
        state: StateDict,
        gen_count: Optional[int] = None,
        split_size: Optional[Union[int, float]] = None,
        train_plugin: Optional[str] = None,
        do_eval: bool = True,
    ) -> List[StateDict]:
        gen_args = dict(self.gen_args)
        if gen_count is None:
            if gen_args.get("count", None) is None:
                gen_args["count"] = 1 if state.train is None else len(state.train)
        else:
            gen_args["count"] = gen_count
        if isinstance(gen_args["count"], list):
            state_dicts = (self._call(state.clone(), gen_count=c) for c in gen_args["count"]) #TODO: parallelize this
            return list(chain.from_iterable(state_dicts))
        split_args = dict(self.split_args)
        if split_size is None:
            if split_args.get("train_size", None) is None:
                if state.test is None:
                    split_args["size"] = 0.8
        else:
            split_args["size"] = split_size
        if "size" in split_args and isinstance(split_args["size"], list):
            state_dicts = (self._call(state.clone(), split_size=s) for s in split_args["size"]) #TODO: parallelize this
            return list(chain.from_iterable(state_dicts))
        train_args = dict(self.train_args)
        if train_plugin is not None:
            train_args["plugin"] = train_plugin
        if "plugin" in train_args and isinstance(train_args["plugin"], list):
            state_dicts = (self._call(state.clone(), plugin=p) for p in train_args["plugin"]) #TODO: parallelize this
            return list(chain.from_iterable(state_dicts))
        if split_args.get("size", None) is not None:
            split_args["train_size"] = split_args["size"]
            del split_args["size"]
            train = self.ensure_output_format(state.train, DataFrame)
            train, test = train_test_split(
                train,
                **split_args,
            )
            state.train = train.reset_index(drop=True)
            state.test = test.reset_index(drop=True)
        state.model = self.train_adapter.train_model(
            data=state.train,
            **self.train_args,
        )
        state.synth = self.train_adapter.generate_data(
            model=state.model,
            **gen_args,
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
        return [state]

class TabularSynthesisDPPipeline(TabularSynthesisPipeline):
    pass
