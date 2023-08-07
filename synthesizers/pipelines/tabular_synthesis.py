from .base import Pipeline

class TabularSynthesisPipeline(Pipeline):
    def __call__(
        self,
        data: object,
        count: int = None,
    ):
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
        if self.output_format is "auto":
            output = self.ensure_output_format(output, output_format=type(data))
        elif self.output_format is not None:
            output = self.ensure_output_format(output)
        return output

class TabularSynthesisDPPipeline(TabularSynthesisPipeline):
    pass