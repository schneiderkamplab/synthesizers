from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Optional, Union

from .base import Pipeline
from ..utils.formats import StateDict

class TabularSplitPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
        size: Optional[Union[int, float]] = None,
    ):
        state = StateDict.wrap(state)
        kwargs = dict(self.kwargs)
        kwargs.update(self.split_args)
        if size is None:
            if kwargs.get("size", None) is None:
                kwargs["size"] = 0.8
        else:
            kwargs["size"] = size
        kwargs["train_size"] = kwargs["size"]
        del kwargs["size"]
        train = self.ensure_output_format(state.train, DataFrame)
        train, test = train_test_split(
            train,
            **kwargs,
        )
        state.train = train.reset_index(drop=True)
        state.test = test.reset_index(drop=True)
        if self.output_format is not None:
            state.train = self.ensure_output_format(state.train)
            state.test = self.ensure_output_format(state.test)
        if self.save_args.get("name", None) is not None:
            state.Save(**self.save_args)
        return state
