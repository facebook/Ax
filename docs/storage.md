---
id: storage
title: Storage
---

Ax has extensible support for saving and loading experiments in both JSON and SQL. The former is a good option for users who prefer lightweight, transportable storage, and the latter is better suited to production applications requiring a centralized, high-performance database.

## JSON

### Saving

To save an experiment to JSON, specify the filepath:

```py
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

```py
from ax import load
experiment = load_experiment(filepath)
```

### Customizing

If you add a custom [`Metric`](/api/core.html#module-ax.core.metric) or [`Runner`](../api/core.html#ax.core.runner.Runner) and want to ensure it is saved to JSON properly, simply call [`register_metric`](/api/data.html#.data.users.adamobeng.fbsource.fbcode.ax.ax.storage.metric_registry.register_metric) or [`register_runner`](/api/data.html#.data.users.adamobeng.fbsource.fbcode.ax.ax.storage.runner_registry.register_runner):

```py
from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner

class MyRunner(Runner):
    pass

class MyMetric(Metric):
    pass

register_metric(MyMetric)
register_runner(MyRunner)
```

## SQL

### Saving

To save an experiment to SQL, first initialize a session by passing a URL pointing to your database. Such a URL is typically composed of a dialect (e.g. sqlite, mysql, postgresql), optional driver (DBAPI used to connect to the database; e.g. psycopg2 for postgresql), username, password, hostname, and database name. A more detailed explanation how to generate a URL can be found in the [SQLAlchemy docs](https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls).

```py
from ax.storage.sqa_store.db import init_engine_and_session_factory

# url is of the form "dialect+driver://username:password@host:port/database"
init_engine_and_session_factory(url="postgresql+psycopg2://sarah:c82i94d@localhost:5432/foobar")
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

If you add a custom [`Metric`](/api/core.html#module-ax.core.metric) or [`Runner`](../api/core.html#ax.core.runner.Runner) and want to ensure it is saved to SQL properly, simply call [`register_metric`](/api/data.html#.data.users.adamobeng.fbsource.fbcode.ax.ax.storage.metric_registry.register_metric) or [`register_runner`](/api/data.html#.data.users.adamobeng.fbsource.fbcode.ax.ax.storage.runner_registry.register_runner):

```py
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
