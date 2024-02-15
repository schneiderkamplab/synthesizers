# synthesizers
A meta library for synthetic data generation.

The goal of synthesizers is to simnplify the use of existing frameworks for synthethic data generation:
* all basic operations are available as pipelines that transform states
* states keep track of datasets, models, and evaluation results
* a meta pipeline allows for very simple but expressive synthetic data generation
* datasets are read from CSV, JSON, JSONL, and Excel (.xlsx) files
* datasets can be downloaded from the Huggingface Hub
* states including datasets and models can be saved and loaded from disk
* datasets can be converted between list, Numpy, Pandas, and Huggingface datasets formats
* datasets are automatically converted to the input format of synthesis or evaluation backends

Meta pipeline usage example, which loads a dataset from Huggingface Hub, trains a GAN model, synthesizes a 10,000 rows synthetic version, and saves the synthetic data in an Excel file:
```
from synthesizers import pipeline
pipeline("synthesize", gen_count=10000, eval_target_col="is_cancer")("mstzt/breast").save("breast.xlsx", key="synth")
```

The same operations with base pipelines:
```
from synthesizers import pipeline
state = pipeline("train")("mstz/breast")
state = pipeline("generate", count=10000)(state)
state = pipeline("evaluate", target_col="is_cancer")
state.save("breast.xlsx", key="synth")

```

# Development TODOs
* split pipeline (part of meta pipeline with split_ prefix, make shuffle and seed parameters)
* implement save pipeline
* update tests and test notebooks to reflect 1.0.0
* have the id pipeline available in __init__.py as load instead of loader
* implement functional versions for all pipelines
* wait for syntheval to fix target_col==None and release 1.0.0 afterwards

# Ideas for future development
* use benchmark module from syntheval?
* standardized list of supported metrics (supported by any backend)
* standardized list of supported generation methods (supported by any backend)
* accumulation of multiple outputs (model, synth, and eval as lists)
* select and combine evaluation backends automatically for given list of metrics
* select generation backend automatically for given generation method
* make syntheval plots available as PIL images
* push_to_hub method on models a la https://github.com/huggingface/datasets/blob/main/src/datasets/arrow_dataset.py
* push_to_hub method on datasets
* R synthpop as backend
* integration of other backends
