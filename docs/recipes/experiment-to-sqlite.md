# Saving and Loading an Ax Experiment to SQLite

Ax provides a convenient way to save and load experiments in SQLite format, making it easy to store and manage experiment data. In this recipe, we will walk through the steps of saving and loading an Ax experiment to SQLite.

## Introduction

Saving an experiment to SQLite is useful when you want to store the experiment data in a structured and queryable format. This can be particularly useful for production applications that require a centralized, high-performance database.

## Prerequisites

We assume that you are already familiar with using Ax for experimentation and have an AxClient instance set up.

## Setup

Before we begin, make sure you have an AxClient instance configured for your experiment.

```python
from ax import AxClient

client = AxClient()
```

## Steps

1. Initialize a session by passing a URL pointing to your SQLite database
2. Create all tables in the database
3. Save the experiment to SQLite
4. Load the experiment from SQLite

### 1. Initialize a session by passing a URL pointing to your SQLite database

First, we need to initialize a session by passing a URL pointing to your SQLite database. Such a URL is typically composed of a dialect (e.g. sqlite, mysql, postgresql), optional driver (DBAPI used to connect to the database; e.g. psycopg2 for postgresql), username, password, hostname, and database name.

 A more detailed explanatino on how to generate a URL can be found in the [SQLAlchemy docs](https://docs.sqlalchemy.org/en/13/core/engines.html?fbclid=IwZXh0bgNhZW0CMTEAAR2abpGB86CC2iA9ZgXltBODU-tHfP_cvlXay2opsGUI70GKv2I7q2UZPDY_aem_N5fQ7unkL8WLcsbaftrQuQ#database-urls)

```python
from ax.storage.sqa_store.db import init_engine_and_session_factory

# url is of the form "dialect+driver://username:password@host:port/database"
url = "postgresql+psycopg2://[USERNAME]:[PASSWORD]@localhost:[PORT]/[DATABASE]"
init_engine_and_session_factory(url=url)
```

### 2. Create all tables in the database

Next, we need to create all tables in the database. We can do this by calling the `create_all_tables` function and passing the engine instance.

```python
from ax.storage.sqa_store.db import get_engine, create_all_tables

engine = get_engine()
create_all_tables(engine)
```

### 3. Save the experiment to SQLite

Now, we can save the experiment to SQLite by calling the `save_experiment` function and passing the experiment instance.

```python
from ax import Experiment
from ax.storage.sqa_store.save import save_experiment

experiment = client._experiment
save_experiment(experiment)
```

Alternatively, you can pass a creator function instead of a url to init_engine_and_session_factory:

```python
from ax import Experiment
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(creator=creator)
experiment = client._experiment
save_experiment(experiment)
```

### 4. Updating the experiment in SQLite
To update a SQLite-backed experiment, simply re-save the experiment to SQLite.

### 5. Load the experiment from SQLite

To load the experiment from SQLite, we can call the `load_experiment` function and pass the experiment name.

```python
from ax import Experiment
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.load import load_experiment

init_engine_and_session_factory(url=dialect+driver://username:password@host:port/database)
loaded_experiment = load_experiment(experiment_name=experiment.name)
```

## Customizing

### Adding a new metric or runner

If you have custom metrics or runners that you want to ensure are saved to SQLite properly, you can create a `RegistryBundle` that bundles together encoding and decoding logic for use in the various save/load functions.

```python
from ax.storage.registry_bundle import RegistryBundle
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.sqa_config import SQAConfig

bundle = RegistryBundle(
    metric_clss={MyMetric: None},
    runner_clss={MyRunner: None}
)

sqa_config = SQAConfig(
    json_encoder_registry=bundle.encoder_registry,
    json_decoder_registry=bundle.decoder_registry,
    metric_registry=bundle.metric_registry,
    runner_registry=bundle.runner_registry,
)

save_experiment(experiment, config=sqa_config)
loaded_experiment = load_experiment(experiment_name=experiment.name, config=sqa_config)
```

### Specifying experiment types:

If you choose to add types to your experiments, create an Enum mapping experiment types to integer representations, pass this Enum to a custom instance of SQAConfig, and then pass the config to sqa_store.save:

```python
from ax import Experiment
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.sqa_config import SQAConfig
from enum import Enum

class ExperimentType(Enum):
    DEFAULT: 0

config = SQAConfig(experiment_type_enum=ExperimentType)
save_experiment(experiment, config=config)
```

### Specifying generator run types:

If you choose to add types to your generator runs (beyond the existing status_quo type), create an enum mapping generator run types to integer representations, pass this enum to a custom instance of SQAConfig, and then pass the config to sqa_store.save:

```python
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


## Learn more

Take a look at these other recipes to continue your learning:

- [Saving and Loading an Ax Experiment to JSON](#)
