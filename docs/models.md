---
id: models
title: Models
---

## Using models in Ax

Describe model factory, get_sobol, get_GPEI, get_empirical_Bayes_thompson, get_factorial. Plots, cross validation. Brief descriptions of transforms and when to use each.


## Deeper dive: Organization of the modeling stack

Ax uses a bridge design to provide a unified interface for models, while still allowing for modularity in how different types of models are implemented. The modeling stack consists of two layers: the ModelBridge and the Model.

The ModelBridge is the object that is directly used in Ax: model factories return ModelBridge objects, and plotting and cross validation tools operate on a ModelBridge. The ModelBridge defines a unified API for all of the models used in Ax via methods like `predict` and `gen`. Internally, it is responsible for transforming AE objects like Arm and Data into objects which are then consumed downstream by a Model.

Model objects are only used in Ax via a ModelBridge. Each Model object defines an API that does not include Ax objects, which allows for modularity of different model types and makes it easy to implement new models. For example, the TorchModel defines an API for a model that operates on torch tensors. There is a 1-to-1 link between ModelBridge objects and Model objects: the TorchModelBridge takes in Ax objects, converts them to torch tensors, and sends them along to the TorchModel. Similar pairings exist for all of the different model types:

| ModelBridge         | Model         | Example implementation        |
| ------------------- | ------------- | ----------------------------- |
| TorchModelBridge    | TorchModel    | BotorchModel                  |
| NumpyModelBridge    | NumpyModel    | TBD                           |
| DiscreteModelBridge | DiscreteModel | EmpiricalBayesThompsonSampler |
| RandomModelBridge   | RandomModel   | SobolGenerator                |

This structure allows for different models like the GP in BotorchModel and the ?? in TBD to share an interface and use common plotting tools at the level of the ModelBridge, while each is implemented using its own torch or numpy structures.

The primary role of the ModelBridge is to act as a transformation layer. This includes transformations to the data, search space, and optimization config such as standardization and log transforms, as well as the final transform from Ax objects into the objects consumed by the Model. We now describe how transforms are implemented and used in the ModelBridge.


## Transforms

Describe how transforms work, how to pass in options to them, point to API reference for full list, describe transforms for each factory function, and describe what needs to be done to implement new transforms. An important detail we should note is that the order in which transforms are applied matters.


## Implementing new models

The structure of the modeling stack makes it easy to implement new models and use them inside Ax. There are two ways this might be done.

### Using an existing Model interface

The easiest way to implement a new model is if it can be adapted to the one of the existing Model interfaces (`TorchModel`, `NumpyModel`, `DiscreteModel`, or `RandomModel`). The class definition provides the interface for each of the methods that should be implemented in order for Ax to be able to fully use the new model. Note however that not all methods must necessarily be implemented in order to use some Ax functionality. An implementation of NumpyModel that implements only `fit` and `predict` can be used to fit data and make plots in Ax; however it will not be able to generate new candidates (requires implementing `gen`) or be used with Ax's cross validation utility (requires implementing `cross_validate`).

Once the new model has been implemented, it can be used in Ax with the corresponding ModelBridge from the table above. For instance, suppose a new numpy-based model was implemented as a subclass of NumpyModel. We can use that model in Ax like:
```
new_model_obj = NewModel(init_args)  # An instance of the new model class
m = NumpyModelBridge(
    experiment=experiment,
    search_space=search_space,
    data=data,
    model=new_model_obj,
    transforms=[UnitX, StandardizeY],
)
```
The ModelBridge object m can then be used with plotting and cross validation utilities exactly the same way as the built-in models.

### Creating a new Model interface

If none of the existing Model interfaces work are suitable for the new model type, then a new interface will have to be created. This involves two steps: creating the new model interface and creating the new model bridge. The new model bridge must be a subclass of ModelBridge  that implements ModelBridge._fit,  ModelBridge._predict, ModelBridge._gen, and  ModelBridge._cross_validate. The implementation of each of these methods will transform the Ax objects in the inputs into objects required for the interface with the new model type. The model bridge will then call out to the new model interface to do the actual modeling work. All of the ModelBridge/Model pairs in the table above provide examples of how this interface can be defined. The main key is that the inputs on the ModelBridge side are fixed, but those inputs can then be transformed in whatever way is desired for the downstream Model interface to be that which is most convenient for implementing the model.
