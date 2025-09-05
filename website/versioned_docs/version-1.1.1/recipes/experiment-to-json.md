# Saving and Loading an Ax Experiment to JSON

Ax provides a convenient way to save and load experiments in JSON format, making
it easy to store and transport experiment data. In this recipe, we will walk
through the steps of saving and loading an Ax experiment to JSON using the
Client.

## Introduction

Saving and loading an experiment to JSON is useful when you want to store the
experiment data in a lightweight and transportable format. This can be
particularly useful for users who prefer a simple storage solution or need to
share experiment data with others.

## Steps

In this recipe, we plan to save a a snapshot of an Ax Client to JSON, reload the
client from the JSON file, and validate its contents.

- **Saving to JSON**
  1. Initialize a Client and configure it with an experiment
  2. View the state of the attached experiment via `summarize()`
  3. Call save_to_json_file to save a snapshot of the Client to JSON
- **Loading from JSON**
  1. Call load_from_json_file to initialize a new Client
  2. Validate the state of the attached experiment via `summarize()`

### Saving to JSON

#### 1. Initialize a Client and configure it with an experiment

Instantiate a `Client` and configure it for your experiment.

```python
client = Client()

client.configure_experiment(...)
client.configure_optimization(...)
```

#### 2. View the state of the attached experiment via `summarize()`

You can inspect the state of a experiment by leveraging the summarize() method,
which returns a DataFrame

```python
client.summarize()
```

#### 2. Call save_to_json_file to save a snapshot of the Client to JSON

In order to save an experiment to JSON, we need to call the `save_to_json_file`
method on the Client instance. This method takes a single optional argument
`filepath`, which is the filepath where we want to save the JSON file (argument
defaults to "ax_client_snapshot.json").

```python
client.save_to_json_file()
```

On success, this will save a snapshot of the Client's settings and state to the
specified file, which includes information about the experiment and generation
strategy (if present).

### Load an Experiment from JSON

#### 1. Call load_from_json_file to initialize a new Client

We will now load the previously saved Client snapshot into a new one. We will do
this by calling `load_from_json_file`

```python
new_client = Client.load_from_json_file(filepath = "ax_client_snapshot.json")
```

#### 2. Validate the state of the attached experiment via `summarize()`

We can now view the state of the experiment by calling `summarize()`, and
validate that it is the same as the one we saved in the earlier section

```python
new_client.summarize()
```

### Customizing the Serialization Process

If you have custom metrics or runners that you want to ensure are saved to JSON
properly, you can initialize the `Client` with a `StorageConfig` that contains a
`RegistryBundle`, that bundles together encoding and decoding logic for use in
the save/load functions.

```python
storage_config = StorageConfig(
    registry_bundle = RegistryBundle(
        runner_clss={MyRunner: None},
        metric_clss={MyMetric: None},
    )
)

client = Client(storage_config = storage_config)
```

## Learn more

Take a look at these other recipes to continue your learning:

- [Saving, Loading, and Updating an Ax Experiment from SQLite](../recipes/experiment-to-sqlite.md)
