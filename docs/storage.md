---
id: storage
title: Storage
---

Ax has extensible support for saving and loading experiments in both JSON and MySQL. The former is a good option for users who prefer lightweight, transportable storage, and the latter is better suited to production applications requiring a centralized, high-performance database.

## JSON

### Saving

To save an experiment to JSON, specify the filepath:

```
from ax import Experiment, save

experiment = Experiment(...)
filepath = "experiments/experiment.json"
save(experiment, filepath)
```

The experiment (including attached data) will be serialized and saved to the specified file.

### Updating

To update a JSON-backed experiment, re-save to the same file.

### Loading

To load an experiment from JSON, specify the filepath again:

```
from ax import load
experiment = load_experiment(filepath)
```

### Customizing

If you add a custom Metric or Runner and want to ensure it is saved to JSON properly, simply call `register_metric` or `register_runner`:

```
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner

class MyRunner(Runner):
    def run():
        pass

class MyMetric(Metric):
    pass

register_metric(MyMetric)
register_runner(MyRunner)
```

## MySQL

### Saving

To save an experiment to MySQL, first initialize a session by passing a url pointing to your database:

```
from ax.storage.sqa_store.db import init_engine_and_session_factory

init_engine_and_session_factory(url=dialect+driver://username:password@host:port/database)
```

Then create all tables:
```
from ax.storage.sqa_store.db import get_engine, create_all_tables

engine = get_engine()
create_all_tables(engine)
```

Then save your experiment:
```
from ax import Experiment
from ax.sqa_store import save

experiment = Experiment(...)
save(experiment)
```

The experiment (including attached data) will be converted to SQLAlchemy classes and saved to the corresponding tables.

Alternatively, you can pass a [creator function](https://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine.params.creator) instead of a url to `init_engine_and_session_factory`:

```
from ax import Experiment, sqa_store
from ax.storage.sqa_store.db import init_engine_and_session_factory

init_engine_and_session_factory(creator=creator)
experiment = Experiment(...)
sqa_store.save(experiment)
```

### Updating

To update a MySQL-backed experiment, call `save_experiment(experiment)` again. The library will re-convert the experiment to SQLAlchemy classes, compare the new instances to the existing ones, and perform any necessary updates.

### Loading

To load an experiment from MySQL, specify the name:

```
from ax import Experiment, sqa_store
from ax.storage.sqa_store.db import init_engine_and_session_factory

init_engine_and_session_factory(url=dialect+driver://username:password@host:port/database)
experiment = sqa_store.load(experiment_name)
```

### Customizing

**Adding a new metric or runner:**

If you add a custom Metric or Runner and want to ensure it is saved to JSON properly, simply call `register_metric` or `register_runner`:

```
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner

class MyRunner(Runner):
    def run():
        pass

class MyMetric(Metric):
    pass

register_metric(MyMetric)
register_runner(MyRunner)
```

**Custom storage functionality:**

1. Instantiate an instance of `SQAConfig` (defined in `ax.storage.sqa_store.sqa_config`) with custom parameters (see the sections below for more details).
2. Instantiating instances of `Encoder` (defined in `ax.storage.sqa_store.encoder`) and `Decoder` (defined in `ax.storage.sqa_store.base_decoder`) with your custom config.
3. Pass your custom `Encoder` and `Decoder` into the `save_experiment_with_encoder` and `load_experiment_with_encoder` methods.

```
config = SQAConfig(...)
encoder = Encoder(config=config)
decoder = Decoder(config=config)

save_experiment_with_encoder(experiment, encoder=encoder)
load_experiment_with_decoder(experiment, decoder=decoder)
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
