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
from .identity import IdentityPipeline
from .tabular_evaluation import TabularEvaluationPipeline
from .tabular_generation import TabularGenerationPipeline
from .tabular_split import TabularSplitPipeline
from .tabular_synthesis import TabularSynthesisPipeline, TabularSynthesisDPPipeline
from .tabular_training import TabularTrainingPipeline

# Register all the supported tasks here
TASK_ALIASES = {
    "evaluate": "tabular-evaluation",
    "generate": "tabular-generation",
    "id": "identity",
    "split": "tabular-split",
    "synthesize": "tabular-synthesis",
    "train": "tabular-training",
}
SUPPORTED_TASKS = {
    "identity": {
        "impl": IdentityPipeline,
        "train_adapter": None,
        "eval_adapter": None,
        "type": None,
    },
    "tabular-evaluation": {
        "impl": TabularEvaluationPipeline,
        "train_adapter": None,
        "eval_adapter": SynthEvalAdapter(),
        "type": "tabular",
    },
    "tabular-generation": {
        "impl": TabularGenerationPipeline,
        "train_adapter": None,
        "eval_adapter": None,
        "type": "tabular",
    },
    "tabular-split": {
        "impl": TabularSplitPipeline,
        "train_adapter": None,
        "eval_adapter": None,
        "type": "tabular",
    },
    "tabular-synthesis": {
        "impl": TabularSynthesisPipeline,
        "train_adapter": SynthCityAdapter(),
        "eval_adapter": SynthEvalAdapter(),
        "type": "tabular",
    },
    # This task is a special case as it's parametrized by EPSILON and DELTA.
    "tabular-synthesis-dp": {
        "impl": TabularSynthesisDPPipeline,
        "train_adapter": SynthCityAdapter(plugin="dpgan"),
        "eval_adapter": SynthEvalAdapter(),
        "type": "tabular",
    },
    "tabular-training": {
        "impl": TabularTrainingPipeline,
        "train_adapter": SynthCityAdapter(),
        "eval_adapter": None,
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
    train_adapter: Optional[Union[Adapter, str]] = None,
    eval_adapter: Optional[Union[Adapter, str]] = None,
    pipeline_class: Optional[Type] = None,
    **kwargs,
):
    normalized_task, targeted_task, task_options = check_task(task)
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]
    if train_adapter is None:
        train_adapter = targeted_task["train_adapter"]
    if eval_adapter is None:
        eval_adapter = targeted_task["eval_adapter"]
    train_args = {k.split("train_")[1]: v for k, v in kwargs.items() if k.startswith("train_")}
    gen_args = {k.split("gen_")[1]: v for k, v in kwargs.items() if k.startswith("gen_")}
    eval_args = {k.split("eval_")[1]: v for k, v in kwargs.items() if k.startswith("eval_")}
    split_args = {k.split("split_")[1]: v for k, v in kwargs.items() if k.startswith("split_")}
    save_args = {k.split("save_")[1]: v for k, v in kwargs.items() if k.startswith("save_")}
    kwargs = {k: v for k, v in kwargs.items() if k.split("_")[0] not in ("train", "gen", "eval")}
    return pipeline_class(
        task=task,
        train_adapter=train_adapter,
        eval_adapter=eval_adapter,
        train_args=train_args,
        gen_args=gen_args,
        eval_args=eval_args,
        split_args=split_args,
        save_args=save_args,
        **kwargs,
    )

Load = pipeline("id")