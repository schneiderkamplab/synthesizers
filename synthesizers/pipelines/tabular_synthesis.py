from pandas import DataFrame

from .base import Pipeline
from ..utils import loader

class TabularSynthesisPipeline(Pipeline):
    def __call__(
        self,
        data: object,
        count: int = None,
        do_eval: bool = True,
    ):
        if type(data) == str:
            data = loader(data)
        if count is None:
            # assumption that all formats implement __len__
            count = len(data)
        model = self.adapter.train_model(
            data=data,
            **self.kwargs,
        )
        output = self.adapter.generate_data(
            count=count,
            model=model,
            **self.kwargs,
        )
        if do_eval:
            self.adapter.evaluate_generated(
                original_data=data,
                generated_data=output,
                **self.kwargs,
            )
        if self.output_format is "auto" and type(data) != str:
            output = self.ensure_output_format(output, output_format=type(data))
        elif self.output_format is not None:
            output = self.ensure_output_format(output)
        return output

class TabularSynthesisDPPipeline(TabularSynthesisPipeline):
    pass