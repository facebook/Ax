---
id: storage
title: Storage
---

Ax has extensible support for saving and loading experiments in both JSON and MySQL. The former is a good option for users who prefer lightweight, transportable storage, and the latter is better suited to production applications requiring a centralized, high-performance database.

## JSON

### Saving

To save an experiment to JSON, specify the filepath:

```
from ae.core.experiment import Experiment
from ae.storage.json_store.save import save_experiment

experiment = Experiment(...)
filepath = "experiments/experiment.json"
save_experiment(experiment, filepath)
```

The experiment (including attached data) will be serialized and saved to the specified file.

### Updating

To update a JSON-backed experiment, re-save to the same file.

### Loading

To load an experiment from JSON, specify the filepath again:

```
from ae.storage.json_store.load import load_experiment
experiment = load_experiment(filepath)
```

### Customizing

If you add a custom class to Ax (e.g. a metric or runner) and want to ensure it is saved to JSON properly, you'll need to:

1. Add a `class_to_dict` method in `ae/storage/json_store/encoders.py`. This method should return a dictionary indicating which fields of the object to save. The dictionary should include a `__type` field with the name of the class (to be used during deserialization). You do not need to manually serialize values of the fields in the dictionary in this method; each value will be be recursively serialized via the `object_to_json` method in `ae.storage.json_store.encoder`.
2. Add your `class_to_dict` method to `ENCODER_REGISTRY` in `ae.storage.json_store.registry`, and add your class to `DECODER_REGISTRY`.

## MySQL

### Saving

To save an experiment to MySQL, you must first setup a database and create all necessary tables.

[TODO: Script for this]

Then, initialize a session by passing a url pointing to your database:

```
from ae.core.experiment import Experiment
from ae.storage.sqa_store.db import init_engine_and_session_factory
from ae.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(url=dialect+driver://username:password@host:port/database)
experiment = Experiment(...)
save_experiment(experiment)
```

The experiment (including attached data) will be converted to SQLAlchemy classes and saved to the corresponding tables.

Alternatively, you can pass a [creator function](https://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine.params.creator) instead of a url to `init_engine_and_session_factory`:

```
from ae.core.experiment import Experiment
from ae.storage.sqa_store.db import init_engine_and_session_factory
from ae.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(creator=creator)
experiment = Experiment(...)
save_experiment(experiment)
```

### Updating

To update a MySQL-backed experiment, call `save_experiment(experiment)` again. The library will re-convert the experiment to SQLAlchemy classes, compare the new instances to the existing ones, and perform any necessary updates.

### Loading

To load an experiment from MySQL, specify the name:

```
from ae.core.experiment import Experiment
from ae.storage.sqa_store.db import init_engine_and_session_factory
from ae.storage.sqa_store.load import load_experiment

init_engine_and_session_factory(url=dialect+driver://username:password@host:port/database)
experiment = load_experiment(experiment_name)
```

### Customizing

You can add custom storage functionality as follows:

1. Instantiate an instance of `SQAConfig` (defined in `ae.storage.sqa_store.sqa_config`) with custom parameters (see the sections below for more details).
2. Instantiating instances of `Encoder` (defined in `ae.storage.sqa_store.encoder`) and `Decoder` (defined in `ae.storage.sqa_store.base_decoder`) with your custom config.
3. Pass your custom `Encoder` and `Decoder` into the `save_experiment_with_encoder` and `load_experiment_with_encoder` methods.

```
config = SQAConfig(...)
encoder = Encoder(config=config)
decoder = Decoder(config=config)

save_experiment_with_encoder(experiment, encoder=encoder)
load_experiment_with_decoder(experiment, decoder=decoder)
```

**Adding a new metric or runner:**

If you add a custom metric (or runner) to Ax, instantiate an instance of `MetricRegistry` (`RunnerRegistry`), defined in `ae/metrics/registry.py `(`ae/runners/registry.py`) with a `class_to_type` dictionary that includes a mapping from the new class to an integer representation. Pass this registry to `SQAConfig`:

```
class MyMetric(Metric):
    ...

metric_registry = MetricRegistry(class_to_type={MyMetric: 0})
config = SQAConfig(metric_registry=metric_registry)
```

**Specifying experiment types:**

If you choose to add types to your experiments, create an enum mapping experiment types to integer representations, and pass this enum to `SQAConfig`:


```
from enum import Enum

class ExperimentType(Enum):
    DEFAULT: 0

config = SQAConfig(experiment_type_enum=ExperimentType)
```

**Specifying generator run types:**

If you choose to add types to your generator runs (beyond the existing `status_quo` type), create an enum mapping generator run types to integer representations, and pass this enum to `SQAConfig`:


```
from enum import Enum

class GeneratorRunType(Enum):
    DEFAULT: 0
    STATUS_QUO: 1

config = SQAConfig(generator_run_type_enum=GeneratorRunType)
```
