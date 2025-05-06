# Saving and Loading an Ax Experiment to Your SQL Database

Ax provides a convenient way to save and load experiments in SQL format, making
it easy to store and manage experiment data. In this recipe, we will walk
through the steps of saving and loading an Ax experiment to/from your SQL
database.

## Introduction

Saving and loading an experiment to/from your SQL database is useful when you
want to store the experiment data in a structured and queryable format. This can
be particularly useful for production applications that require a centralized,
high-performance database.

The Ax Client automatically saves experiment data at every stage, including
configuration and metric setup. This eliminates the need for manual saving and
allows you to easily load your experiment data back into a Client whenever
needed.

## Steps

In this recipe, we plan to save a snapshot of an Ax Client to a SQL database,
reload the client from the SQL database instance, and validate its contents.

- **Saving to a SQL database**
  1. Initialize a Client with a StorageConfig, and configure it with an
     experiment
  2. View the state of the attached experiment via `summarize()`
- **Loading from a SQL database**
  1. Call load_from_database to initialize a new Client
  2. Validate the state of the attached experiment via `summarize()`

### Saving to a SQL database

#### 1. Initialize a Client with a StorageConfig, and configure it with an experiment

Instantiate a `Client` with a StorageConfig and configure it for your
experiment. In order to initialize a StorageConfig, we need a URL pointing to
your SQL database. Such a URL is typically composed of a dialect (e.g. sqlite,
mysql, postgresql), optional driver (DBAPI used to connect to the database; e.g.
psycopg2 for postgresql), username, password, hostname, and database name.

A more detailed explanation on how to generate a URL can be found in the
[SQLAlchemy docs](https://docs.sqlalchemy.org/en/13/core/engines.html?fbclid=IwZXh0bgNhZW0CMTEAAR2abpGB86CC2iA9ZgXltBODU-tHfP_cvlXay2opsGUI70GKv2I7q2UZPDY_aem_N5fQ7unkL8WLcsbaftrQuQ#database-urls)

```python
client = Client()

url = "sqlite:///path/to/database.db"
storage_config = StorageConfig(url = url)

client.configure_experiment(...)
client.configure_optimization(...)
```

#### 2. View the state of the attached experiment via `summarize()`

You can inspect the state of the experiment by leveraging the summarize()
method, which returns a DataFrame

```python
client.summarize()
```

### Load an Experiment from a SQL database

#### 1. Call load_from_database to initialize a new Client

We will now load the previously saved Client snapshot into a new one. We will do
this by calling `load_from_database`, and passing it the experiment name and
StorageConfig we created in the previous section

```python
new_client = Client.load_from_database(
    experiment_name = experiment_name,
    storage_config = storage_config
)
```

#### 2. Validate the state of the attached experiment via `summarize()`

We can now view the state of the experiment by calling `summarize()`, and
validate that it is the same as the one we saved in the earlier section

```python
new_client.summarize()
```

### Customizing the Serialization Process

If you have custom metrics or runners that you want to ensure are saved to your
SQL database properly, you can initialize the `Client` with a `StorageConfig`
that contains a `RegistryBundle`, that bundles together encoding and decoding
logic for use in the save/load functions.

```python
storage_config = StorageConfig(
    registry_bundle = RegistryBundle(
        runner_clss={MyRunner: None},
        metric_clss={MyMetric: None},
    )
)

client = Client(
    url = url,
    storage_config = storage_config)
```

## Learn more

Take a look at these other recipes to continue your learning:

- [Saving and Loading an Ax Experiment to JSON](../recipes/experiment-to-json.md)
