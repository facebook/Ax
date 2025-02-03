---
id: storage
title: Storage
---
Ax has extensible support for saving and loading experiments in both JSON and SQL. The former is a good option for users who prefer lightweight, transportable storage, and the latter is better suited to production applications requiring a centralized, high-performance database.

## JSON

### Saving

To save an experiment to JSON, specify the filepath:

```py
from ax import Experiment
from ax.storage.json_store.save import save_experiment

experiment = Experiment(...)
filepath = "experiments/experiment.json"
save_experiment(experiment, filepath)
```

The experiment (including attached data) will be serialized and saved to the specified file.

### Updating

To update a JSON-backed experiment, re-save to the same file.

### Loading

To load an experiment from JSON, specify the filepath again:

```py
from ax.storage.json_store.load import load_experiment
experiment = load_experiment(filepath)
```

### Customizing

If you add a custom [`Metric`](https://ax.readthedocs.io/en/latest/core.html#module-ax.core.metric) or [`Runner`](https://ax.readthedocs.io/en/latest/core.html#ax.core.runner.Runner) and want to ensure it is saved to JSON properly, create a [`RegistryBundle`](https://ax.readthedocs.io/en/latest/storage.html#ax.storage.registry_bundle.RegistryBundle), which bundles together encoding and decoding logic for use in the various save/load functions as follows:

```py
from ax import Experiment, Metric, Runner, SearchSpace
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from ax.storage.registry_bundle import RegistryBundle

# Minimal custom runner/metric.
class MyRunner(Runner):
    def run():
        pass

class MyMetric(Metric):
    pass

# Minimal experiment must have a search space, plus our custom classes.
experiment = Experiment(
    search_space=SearchSpace(parameters=[]),
    runner=MyRunner(),
    tracking_metrics=[MyMetric(name="my_metric")]
)

# A RegistryBundle allows Ax to encode/decode the custom classes.
bundle = RegistryBundle(
    runner_clss={MyRunner: None}
    metric_clss={MyMetric: None},
)

filepath = "experiments/experiment.json"
save_experiment(experiment=experiment, filepath=filepath, encoder_registry=bundle.encoder_registry)

loaded_experiment=load_experiment(filepath=filepath, decoder_registry=bundle.decoder_registry)
```

## SQL

### Saving

To save an experiment to SQL, first initialize a session by passing a URL pointing to your database. Such a URL is typically composed of a dialect (e.g. sqlite, mysql, postgresql), optional driver (DBAPI used to connect to the database; e.g. psycopg2 for postgresql), username, password, hostname, and database name. A more detailed explanation how to generate a URL can be found in the [SQLAlchemy docs](https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls).

```py
from ax.storage.sqa_store.db import init_engine_and_session_factory

# url is of the form "dialect+driver://username:password@host:port/database"
init_engine_and_session_factory(url="postgresql+psycopg2://[USERNAME]:[PASSWORD]@localhost:[PORT]/[DATABASE]")
```

Then create all tables:

```py
from ax.storage.sqa_store.db import get_engine, create_all_tables

engine = get_engine()
create_all_tables(engine)
```

Then save your experiment:

```py
from ax import Experiment
from ax.storage.sqa_store.save import save_experiment

experiment = Experiment(...)
save_experiment(experiment)
```

The experiment (including attached data) will be saved to the corresponding tables.

Alternatively, you can pass a [creator function](https://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine.params.creator) instead of a url to `init_engine_and_session_factory`:

```py
from ax import Experiment
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(creator=creator)
experiment = Experiment(...)
save_experiment(experiment)
```

### Updating

To update a SQL-backed experiment, call `save_experiment(experiment)` again. Ax will determine what updates to perform.

### Loading

To load an experiment from SQL, specify the name:

```py
from ax import Experiment
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.load import load_experiment

init_engine_and_session_factory(url=dialect+driver://username:password@host:port/database)
experiment = load_experiment(experiment_name)
```

### Customizing

**Adding a new metric or runner:**

If you add a custom [`Metric`](https://ax.readthedocs.io/en/latest/core.html#module-ax.core.metric) or [`Runner`](https://ax.readthedocs.io/en/latest/core.html#ax.core.runner.Runner) and want to ensure it is saved to SQL properly, create a [`RegistryBundle`](https://ax.readthedocs.io/en/latest/storage.html#ax.storage.registry_bundle.RegistryBundle), which bundles together encoding and decoding logic for use in the various save/load functions as follows:

```py
from ax import Experiment, RangeParameter, ParameterType
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.sqa_config import SQAConfig

# Minimal custom runner/metric.
class MyRunner(Runner):
    def run():
        pass

class MyMetric(Metric):
    pass

# Minimal experiment for SQA must have a name and a nonempty SearchSpace, plus our custom classes.
experiment = Experiment(
    name="my_experiment",
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                lower=0,
                upper=1,
                name="my_parameter",
                parameter_type=ParameterType.FLOAT
            )
        ]
    ),
    runner=MyRunner(),
    tracking_metrics=[MyMetric(name="my_metric")],
)

# The RegistryBundle contains our custom classes.
bundle = RegistryBundle(
    metric_clss={MyMetric: None},
    runner_clss={MyRunner: None}
)

# Abstract this into a SQAConfig as follows, to make loading/saving a bit simpler.
sqa_config = SQAConfig(
    json_encoder_registry=bundle.encoder_registry,
    json_decoder_registry=bundle.decoder_registry,
    metric_registry=bundle.metric_registry,
    runner_registry=bundle.runner_registry,
)

save_experiment(experiment, config=sqa_config)

loaded_experiment = load_experiment(experiment_name="my_experiment", config=sqa_config)
```

**Specifying experiment types:**

If you choose to add types to your experiments, create an Enum mapping experiment types to integer representations, pass this Enum to a custom instance of `SQAConfig`, and then pass the config to `sqa_store.save`:

```py
from ax import Experiment
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.sqa_config import SQAConfig
from enum import Enum

class ExperimentType(Enum):
    DEFAULT: 0

config = SQAConfig(experiment_type_enum=ExperimentType)
save_experiment(experiment, config=config)
```

**Specifying generator run types:**

If you choose to add types to your generator runs (beyond the existing `status_quo` type), create an enum mapping generator run types to integer representations, pass this enum to a custom instance of `SQAConfig`, and then pass the config to `sqa_store.save`:

```py
from ax import Experiment
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.sqa_config import SQAConfig
from enum import Enum

class GeneratorRunType(Enum):
    DEFAULT: 0
    STATUS_QUO: 1

config = SQAConfig(generator_run_type_enum=GeneratorRunType)
save_experiment(experiment, config=config)
```
