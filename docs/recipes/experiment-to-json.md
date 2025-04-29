# Saving and Loading an Ax Experiment to JSON

Ax provides a convenient way to save and load experiments in JSON format, making it easy to store and transport experiment data. In this recipe, we will walk through the steps of saving an Ax experiment to JSON using the AxClient.

## Introduction

Saving an experiment to JSON is useful when you want to store the experiment data in a lightweight and transportable format. This can be particularly useful for users who prefer a simple storage solution or need to share experiment data with others.

## Prerequisites

We assume that you are already familiar with using Ax for experimentation and have an AxClient instance set up.

## Setup

Before we begin, make sure you have an AxClient instance configured for your experiment.

```python
from ax import AxClient

client = AxClient()
```

## Steps

1. Get the experiment object from the AxClient
2. Save the experiment to JSON using the `save_experiment` function

### 1. Get the experiment object from the AxClient

First, we need to get the experiment object from the AxClient. We can do this by accessing the `_experiment` attribute of the AxClient instance.

```python
experiment = client._experiment
```

### 2. Save the experiment to JSON

Next, we use the `save_experiment` function from the `ax.storage.json_store.save` module to save the experiment to JSON. We need to specify the filepath where we want to save the experiment.

```python
from ax.storage.json_store.save import save_experiment

filepath = "experiments/experiment.json"
save_experiment(experiment, filepath)
```

This will serialize the experiment (including attached data) and save it to the specified file.

## Updating the Experiment

To update a JSON-backed experiment, simply re-save the experiment to the same file.

## Loading the Experiment

To load an experiment from JSON, use the `load_experiment` function from the `ax.storage.json_store.load` module and specify the filepath again.

```python
from ax.storage.json_store.load import load_experiment

loaded_experiment = load_experiment(filepath)
```

## Customizing the Serialization Process

If you have custom metrics or runners that you want to ensure are saved to JSON properly, you can create a `RegistryBundle` that bundles together encoding and decoding logic for use in the save/load functions.

```python
from ax.storage.registry_bundle import RegistryBundle

bundle = RegistryBundle(
    runner_clss={MyRunner: None},
    metric_clss={MyMetric: None},
)

filepath = "experiments/experiment.json"
save_experiment(experiment, filepath, encoder_registry=bundle.encoder_registry)
loaded_experiment = load_experiment(filepath, decoder_registry=bundle.decoder_registry)
```

## Learn more

Take a look at these other recipes to continue your learning:

- [Saving, Loading, and Updating an Ax Experiment from SQLite](#)
