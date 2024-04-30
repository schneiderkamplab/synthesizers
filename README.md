# Synthesizers
A meta library for synthetic data generation.

The goal of synthesizers is to simnplify the use of existing frameworks for synthethic data generation:
* All basic operations are available as functional and pipeline abstractions that transform states.
* States keep track of datasets, models, and evaluation results.
* A meta pipeline allows for very simple but expressive synthetic data generation.
* Datasets are read from CSV, TSV, JSON, JSONL, Python Pickle (.pickle), and Excel (.xlsx) files.
* Datasets can be downloaded from the Huggingface Hub.
* States including datasets and models can be saved and loaded from disk.
* Datasets can be converted between list, Numpy, Pandas, and Huggingface datasets formats.
* Datasets are automatically converted to the input format of synthesis or evaluation backends.

## Installation

Simply install synthesizers using pip from PyPI:
```
pip install synthesizers
```
If you clone or downloaded the source code, you can also install it from the root directory of the repository:
```
pip install .
```
Or download and install directly from the terminal:
```
pip install https://github.com/schneiderkamplab/synthesizers/archive/refs/heads/main.zip
```

To ensure the right dependencies, it is often preferable to create a virtual environment (here the directory `venv` in the current directory):
```
python -m virtualenv venv
. venv/activate
pip install synthesizers
```

Conda is a popular alternative:
```
conda create -n synthesizers python=3.11
conda activate synthesizers
pip install synthesizers
```

## Usage

### Functional abstraction

The functional abstraction manipulates states that can be initalized by the pre-defined `Load` object and manipulated by functions such as the meta function `Synthesize`:
```python
from synthesizers import Load
Load("mstz/breast").Synthesize(split_size=0.8, gen_count=10000, eval_target_col="is_cancer", save_name="breast.xlsx", save_key="synth")
```
In this case, `Load` loads a dataset on breast cancer fromt the Huggingface Hub, resulting in a state containing just a `train` dataset. This state is then expanded by the `Synthesize` function, which splits the `train` dataset into `train` and `test` datasets, trains a GAN `model`, generates a `synth` dataset, computes `eval` information, and saves the synthetic data to an Excel file.

The meta function `Synthesize` can be broken up into separate functions for the individual steps:
```python
from synthesizers import Load
Load("mstz/breast").Split(size=0.8).Train().Generate(count=10000).Evaluate(target_col="is_cancer").Save(name="breast.xlsx", key="synth")
```
This version can be used to resuse intermediate states, e.g., to generate and save synthetic datasets of different sizes reusing the same trained model:
```python
from synthesizers import Load
state = Load("mstz/breast").Split(size=0.8).Train()
for count in (100, 1000, 10000, 100000):
    state.Generate(count=count).Save(name=f"breast-{count}.csv", key="synth")
```
It is also useful when it is necessary to store the intermediate state to the file system:
```python
from synthesizers import Load
state = Load("mstz/breast").Split(size=0.8).Train().Save("breast_state")
```
The saved state can be loaded and resumed as one might expect:
```python
from synthesizers import Load
Load("breast_state").Generate(count=10000).Save(name="breast.csv", key="synth")
```
The `count` parameter can be a list or another iterable sequence, indicating that multiple synthetic sets be created. The following code will save two synthetic datasets to `breast_1000.csv` and `breast_100000.csv`:
```python
from synthesizers import Load
Load("breast_state").Generate(count=[1000,100000]).Save(name="breast_1000.csv", index=0, key="synth").Save(name="breast_100000.csv", index=1, key="synth")
```
Multiple parameters are also allowed for the `plugin` parameter of `Train` and the `size` parameter of `Split`.

Furthermore, the `Load` function takes either a single dataset or a tuple of such datasets. With the help of the optional `jobs` parameter (with variants `train_jobs`, `eval_jobs` etc.) parameter, the number of concurrent processes can be set. In the following example, we generate synthetic versions of two different splits of two different datasets:
```python
from synthesizers import Load
Load(("mstz/titanic","mstz/breast")).Synthesize(split_size=[0.5,0.8], train_jobs=4, do_eval=False).Save("mstz")
```


### Pipeline abstraction

Internally, the functional abstraction instantiates pipelines to accomplish its functionality. These pipelines can be used as an expressive alternative. Here is a usage example with the synthesis meta pipeline, which again loads the breast cancer dataset from the Huggingface Hub, trains a GAN model, synthesizes 10,000 synthetic records, evaluates it, and saves it as a JSON file:
```python
from synthesizers import pipeline
pipeline("synthesize", split_size=0.8, gen_count=10000, eval_target_col="is_cancer", save_name="breast.json", save_key="synth")("mstz/breast")
```

The meta pipeline pools the functionality of multiple base pipelines. The same functionality as in the above example might be accomplished with base pipelines:
```python
from synthesizers import pipeline
state = pipeline("split", size=0.8)("mstz/breast")
state = pipeline("train")(state)
state = pipeline("generate", count=10000)(state)
state = pipeline("evaluate", target_col="is_cancer")
state = pipeline("identity", save_name="breast.json", save_key="synth")
```

Pipelines are exposed not only as an internal representation but provide the ability to reuse settings, e.g., by having a pipeline for training CTGANs. The following example also illustrates that functional and pipeline abstractions can readily be combined as preferred by the user:
```python
from synthesizers import Load, pipeline
s1 = Load("mstz/breast").Split()
s2 = Load("julien-c/titanic-survival").Split()
train = pipeline("train", plugin="ctgan")
train(s1).Generate(count=1000).Save(name="breast.jsonl", key="synth")
train(s2).Generate(count=1000).Save(name="titanic.jsonl", key="synth")
```

The plugins depend on the backend used. The standard backend for generation is [synthcity](https://github.com/vanderschaarlab/synthcity), which offers a variety of plugins including `adsgan`, `ctgan`, `tvae`, and `bayesian_network`.
For evaluation, the standard backend is [SynthEval](https://github.com/schneiderkamplab/syntheval).

## Ideas for future development
* add possibility to allow methods from multiple backenders by allowing multiple adapters (mapping method name to adapter)
* make sure all parameters can be iterables/sequences where it makes sense (e.g. target_col)
* check argument validity before running pipeline
* improved error handling (e.g. evaluating without synth dataset, training without train dataset etc.)
* add source and meta to StateDict with initial data source and parameters to reproduce
* revamp loading saving to a more useful format, e.g., pickle everything to one file instead of directories
* implement overwrite parameter to State with Load(overwrite=...), three values:
  - copy: add new state if a value would be overwritten
  - overwrite: just overwrite the value
  - raise: raise an error if a value would be overwritten
* implement TabularSynthesisDPPipeline
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
* Put string options as literals so they are more visible in tooltips
* Docstrings for all modules used in the examples
