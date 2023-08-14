from typing import Any, Dict, Optional, Tuple, Type, Union

from ..adapters import (
    SynthCityAdapter,
    SynthEvalAdapter,
)
from ..adapters.base import (
    Adapter,
)
from .base import (
    PipelineRegistry,
)
from .tabular_evaluation import TabularEvaluationPipeline
from .tabular_generation import TabularGenerationPipeline
from .tabular_synthesis import TabularSynthesisPipeline, TabularSynthesisDPPipeline
from .tabular_training import TabularTrainingPipeline

# Register all the supported tasks here
TASK_ALIASES = {
    "evaluate": "tabular-evaluation",
    "generate": "tabular-generation",
    "synthesize": "tabular-synthesis",
    "train": "tabular-training",
}
SUPPORTED_TASKS = {
    "tabular-evaluation": {
        "impl": TabularEvaluationPipeline,
        "adapter": SynthEvalAdapter(),
        "type": "tabular",
    },
    "tabular-generation": {
        "impl": TabularGenerationPipeline,
        "adapter": None,
        "type": "tabular",
    },
    "tabular-synthesis": {
        "impl": TabularSynthesisPipeline,
        "adapter": SynthCityAdapter(),
        "type": "tabular",
    },
    # This task is a special case as it's parametrized by EPSILON and DELTA.
    "tabular-synthesis-dp": {
        "impl": TabularSynthesisDPPipeline,
        "adapter": SynthCityAdapter(plugin="dpgan"),
        "type": "tabular",
    },
    "tabular-training": {
        "impl": TabularTrainingPipeline,
        "adapter": SynthCityAdapter(),
        "type": "tabular",
    },
}

PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)

def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"tabular-synthesis"`
            - `"tabular-synthesis_EPSILON_DELTA"`

    Returns:
        (normalized_task: `str`, task_defaults: `dict`, task_options: (`tuple`, None)) The normalized task name
        (removed alias and options). The actual dictionary required to initialize the pipeline and some extra task
        options for parametrized tasks like "tabular-synthesis_EPSILON_DELTA"


    """
    return PIPELINE_REGISTRY.check_task(task)

def pipeline(
    task: str = None,
    adapter: Optional[Union[Adapter, str]] = None,
    pipeline_class: Optional[Type] = None,
    **kwargs,
):
    normalized_task, targeted_task, task_options = check_task(task)
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]
    if adapter is None:
        adapter = targeted_task["adapter"]

    return pipeline_class(task=task, adapter=adapter, **kwargs)

