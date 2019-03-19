---
id: modeling
title: Modeling
---

## Using models in Ax
Models in Ax provide two main APIs:
* `predict(X):` Predict outcomes for *X* using the model.
* `gen(n_candidates):` Use an acquisition function to generate n_candidates.

The model factory is the quickest way to get up and running with an Ax model. 
The factory provides sensible defaults for hyperparameter configuration for a variety of models, as well
as defining a clear interface for what each model needs. 

### Model Types

#### Continuous space
Describe feature space / brief transforms

get_sobol, (can just say get_uniform is the same but uniform)

get_GPEI,

show plots/cross validation

### Discrete space: (this doesn't make sense because Sobol is also for discrete)**
Describe feature space / brief transforms

get_empirical_Bayes_thompson (get_thompson is same but no EB)
plots

get_factorial


## Deeper dive: Organization of the modeling stack
Describe the ModelBridge. List for each modelbridge what the matching Model implementations are. Show how to instantiate a model outside of the factory. Then segue into each of the layers.

### Transforms
Describe how transforms work, how to pass in options to them, point to API reference for full list, describe transforms for each factory function, and describe what needs to be done to implement new transforms. An important detail we should note is that the order in which transforms are applied matters.

### Implementing new models
Describe what is involved in implementing a new model (choose the most appropriate interface, implement whatever functionality you want). Can point to the soon-to-be-implemented RF as an example of a simple model implementation. Note that if none of the existing Model interfaces are appropriate for the type of model desired, then a new modelbridge will also have to be implemented.
