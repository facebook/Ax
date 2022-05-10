---
id: models
title: Models
---

## Using models in Ax

In the optimization algorithms implemented by Ax, models predict the outcomes of metrics within an experiment evaluated at a parameterization, and are used to predict metrics or suggest new parameterizations for trials. Models in Ax are created using factory functions from the [`ax.modelbridge.factory`](../api/modelbridge.html#module-ax.modelbridge.factory). All of these models share a common API with [`predict()`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.predict) to make predictions at new points and [`gen()`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.gen) to generate new candidates to be tested. There are a variety of models available in the factory; here we describe the usage patterns for the primary model types and show how the various Ax utilities can be used with models.

#### Sobol sequence
The [`get_sobol`](../api/modelbridge.html#ax.modelbridge.factory.get_sobol) function is used to construct a model that produces a quasirandom Sobol sequence when[`gen`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.gen) is called. This code generates a scrambled Sobol sequence of 10 points:
```Python
from ax.modelbridge.factory import get_sobol

m = get_sobol(search_space)
gr = m.gen(n=10)
```
The output of [`gen`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.gen) is a [`GeneratorRun`](../api/core.html#ax.core.generator_run.GeneratorRun) object that contains the generated points, along with metadata about the generation process. The generated arms can be accessed at [`GeneratorRun.arms`](../api/core.html#ax.core.generator_run.GeneratorRun.arms).

Additional arguments can be passed to [`get_sobol`](../api/modelbridge.html#ax.modelbridge.factory.get_sobol) such as `scramble=False` to disable scrambling, and `seed` to set a seed (see [model API](../api/models.html#ax.models.random.sobol.SobolGenerator)).

Sobol sequences are typically used to select initialization points, and this model does not implement [`predict`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.predict). It can be used on search spaces with any combination of discrete and continuous parameters.

#### Gaussian Process with EI
Gaussian Processes (GPs) are used for [Bayesian Optimization](bayesopt.md) in Ax, the [`get_GPEI`](../api/modelbridge.html#ax.modelbridge.factory.get_gpei) function constructs a model that fits a GP to the data, and uses the EI acquisition function to generate new points on calls to [`gen`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.gen). This code fits a GP and generates a batch of 5 points which maximizes EI:
```Python
from ax.modelbridge.factory import get_GPEI

m = get_GPEI(experiment, data)
gr = m.gen(n=5, optimization_config=optimization_config)
```

In contrast to [`get_sobol`](../api/modelbridge.html#ax.modelbridge.factory.get_sobol), the GP requires data and is able to make predictions. We make predictions by constructing a list of [`ObservationFeatures`](../api/core.html#ax.core.observation.ObservationFeatures) objects with the parameter values for which we want predictions:
```Python
from ax.core.observation import ObservationFeatures

obs_feats = [
    ObservationFeatures(parameters={'x1': 3.14, 'x2': 2.72}),
    ObservationFeatures(parameters={'x1': 1.41, 'x2': 1.62}),
]
f, cov = m.predict(obs_feats)
```
The output of [`predict`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.predict) is the mean estimate of each metric and the covariance (across metrics) for each point.

All Ax models that implement [`predict`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.predict) can be used with the built-in plotting utilities, which can produce plots of model predictions on 1-d or 2-d slices of the parameter space:
```Python
from ax.plot.slice import plot_slice
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()
render(plot_slice(
    model=m,
    param_name='x1',  # slice on values of 'x1'
    metric_name='metric_a',
    slice_values={'x2': 7.5},  # Fix at this value for the slice
))
```

<div id="slice" style="width: 100%;"></div>

```Python
from ax.plot.contour import plot_contour

render(plot_contour(
    model=m,
    param_x='x1',
    param_y='x2',
    metric_name='metric_a',
))
```

<div id="contour" style="width: 100%;"></div>

Ax also includes utilities for cross validation to assess model predictive performance. Leave-one-out cross validation can be performed as follows:
```Python
from ax.modelbridge.cross_validation import cross_validate, compute_diagnostics

cv = cross_validate(model)
diagnostics = compute_diagnostics(cv)
```
[`compute_diagnostics`](../api/modelbridge.html#ax.modelbridge.cross_validation.compute_diagnostics) computes a collection of diagnostics of model predictions, such as the correlation between predictions and actual values, and the p-value for a Fisher test of the model's ability to distinguish high values from low. A very useful tool for assessing model performance is to plot the cross validated predictions against the actual observed values:
```Python
from ax.plot.diagnostic import interact_cross_validation

render(interact_cross_validation(cv))
```

<div id="cv" style="width: 100%;"></div>

If the model fits the data well, the values will lie along the diagonal. Poor GP fits tend to produce cross validation plots that are flat with high predictive uncertainty - such fits are unlikely to produce good candidates in [`gen`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.gen).

By default, this model will apply a number of transformations to the feature space, such as one-hot encoding of [`ChoiceParameters`](../api/core.html#ax.core.parameter.ChoiceParameter) and log transformation of [`RangeParameters`](../api/core.html#ax.core.parameter.RangeParameter) which have `log_scale` set to `True`. Transforms are also applied to the observed outcomes, such as standardizing the data for each metric. See [the section below on Transforms](models.md#Transforms) for a description of the default transforms, and how new transforms can be implemented and included.

GPs typically does a good job of modeling continuous parameters ([`RangeParameters`](../api/core.html#ax.core.parameter.RangeParameter)). If the search space contains [`ChoiceParameters`](../api/core.html#ax.core.parameter.ChoiceParameter), they will be one-hot-encoded and the GP fit in the encoded space. A search space with a mix of continuous parameters and [`ChoiceParameters`](../api/core.html#ax.core.parameter.ChoiceParameter) that take a small number of values can be modeled effectively with a GP, but model performance may be poor if there are more than about 20 parameters after one-hot encoding. Cross validation is an effective tool for determining usefulness of the GP on a particular problem.

In discrete spaces where the GP does not predict well, a multi-armed bandit approach is often preferred, and we now discuss the models suitable for that approach.

#### Support for mixed search spaces and categorical variables

The most common way of dealing with categorical variables in Bayesian optimization is to one-hot encode the categories to allow fitting a GP model in a continuous space. In this setting, a categorical variable with categories `["red", "blue", "green"]` is represented by three new variables (one for each category). While this is a convenient choice, it can drastically increase the dimensionality of the search space. In addition, the acquisition function is often optimized in the corresponding continuous space and the final candidate is selected by rounding back to the original space, which may result in selecting sub-optimal points according to the acquisition function.

Our new approach uses separate kernels for the categorical and ordinal (continuous/integer) variables. In particular, we use a kernel of the form: $$k(x, y) = k_\text{cat}(x_\text{cat}, y_\text{cat}) \times k_\text{ord}(x_\text{ord}, y_\text{ord}) + k_\text{cat}(x_\text{cat}, y_\text{cat}) + k_\text{ord}(x_\text{ord}, y_\text{ord})$$
For the ordinal variables we can use a standard kernel such as Matérn-5/2, but for the categorical variables we need a way to compute distances between the different categories. A natural choice is to set the distance is 0 if two categories are equal and 1 otherwise, similar to the idea of Hamming distances. This approach can be combined with the idea automatic relevance determination (ARD) where each categorical variable has its own lengthscale. Rather than optimizing the acquisition function in a continuously relaxed space, we optimize it separately over each combination of the categorical variables. While this is likely to result in better optimization performance, it may lead to slow optimization of the acquisition function when there are many categorical variables.

#### Empirical Bayes and Thompson sampling
For [Bandit optimization](banditopt.md), The [`get_empirical_bayes_thompson`](../api/modelbridge.html#ax.modelbridge.factory.get_empirical_bayes_thompson) factory function returns a model that applies [empirical Bayes shrinkage](banditopt.md#empirical-bayes) to a discrete set of arms, and then uses Thompson sampling to construct a policy with the weight that should be allocated to each arms. Here we apply empirical Bayes to the data and use Thompson sampling to generate a policy that is truncated at `n=10` arms:
```Python
from ax.modelbridge.factory import get_empirical_bayes_thompson

m = get_empirical_bayes_thompson(experiment, data)
gr = m.gen(n=10, optimization_config=optimization_config)
```
The arms and their corresponding weights can be accessed as `gr.arm_weights`.

As with the GP, we can use [`predict`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.predict) to evaluate the model at points of our choosing. However, because this is a purely in-sample model, those points should correspond to arms that were in the data. The model prediction will return the estimate at that point after applying the empirical Bayes shrinkage:
```Python
f, cov = m.predict([ObservationFeatures(parameters={'x1': 3.14, 'x2': 2.72})])
```
We can generate a plot that shows the predictions for each arm with the shrinkage using [`plot_fitted`](../api/plot.html#ax.plot.scatter.plot_fitted), which shows model predictions on all in-sample arms:
```Python
from ax.plot.scatter import plot_fitted

render(plot_fitted(m, metric="metric_a", rel=False))
```

<div id="fitted" style="width: 100%;"></div>

#### Factorial designs

The factory function [`get_factorial`](../api/modelbridge.html#ax.modelbridge.factory.get_factorial) can be used to construct a factorial design on a set of [`ChoiceParameters`](../api/core.html#ax.core.parameter.ChoiceParameter).
```Python
from ax.modelbridge.factory import get_factorial

m = get_factorial(search_space)
gr = m.gen(n=10)
```
Like the Sobol sequence, the factorial model is only used to generate points and does not implement [`predict`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.predict).


## Deeper dive: organization of the modeling stack

Ax uses a bridge design to provide a unified interface for models, while still allowing for modularity in how different types of models are implemented. The modeling stack consists of two layers: the [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) and the Model.

The [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) is the object that is directly used in Ax: model factories return [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) objects, and plotting and cross validation tools operate on a [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge). The [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) defines a unified API for all of the models used in Ax via methods like [`predict`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.predict) and [`gen`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge.gen). Internally, it is responsible for transforming Ax objects like [`Arm`](../api/core.html#ax.core.arm.Arm) and [`Data`](../api/core.html#ax.core.data.Data) into objects which are then consumed downstream by a Model.

Model objects are only used in Ax via a [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge). Each Model object defines an API which does not use Ax objects, allowing for modularity of different model types and making it easy to implement new models. For example, the TorchModel defines an API for a model that operates on torch tensors. There is a 1-to-1 link between [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) objects and Model objects. For instance, the TorchModelBridge takes in Ax objects, converts them to torch tensors, and sends them along to the TorchModel. Similar pairings exist for all of the different model types:

| ModelBridge         | Model         | Example implementation        |
| ------------------- | ------------- | ----------------------------- |
| [`TorchModelBridge`](../api/modelbridge.html#module-ax.modelbridge.torch)    | [`TorchModel`](../api/models.html#ax.models.torch_base.TorchModel)   | [`BotorchModel`](../api/models.html#ax.models.torch.botorch.BotorchModel)                  |           |
| [`DiscreteModelBridge`](../api/modelbridge.html#module-ax.modelbridge.discrete) | [`DiscreteModel`](../api/models.html#ax.models.discrete_base.DiscreteModel) | [`ThompsonSampler`](../api/models.html#ax.models.discrete.thompson.ThompsonSampler)               |
| [`RandomModelBridge`](../api/modelbridge.html#module-ax.modelbridge.random)   | [`RandomModel`](../api/models.html#ax.models.random.base.RandomModel)  | [`SobolGenerator`](../api/models.html#ax.models.random.sobol.SobolGenerator)                |

This structure allows for different models like the GP in BotorchModel and the Random Forest in RandomForest to share an interface and use common plotting tools at the level of the ModelBridge, while each is implemented using its own torch or numpy structures.

The primary role of the [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) is to act as a transformation layer. This includes transformations to the data, search space, and optimization config such as standardization and log transforms, as well as the final transform from Ax objects into the objects consumed by the Model. We now describe how transforms are implemented and used in the ModelBridge.


## Transforms

The transformations in the [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) are done by chaining together a set of individual Transform objects. For continuous space models obtained via factory functions ([`get_sobol`](/api/data.html#.data.users.adamobeng.fbsource.fbcode.ax.ax.modelbridge.factory.get_sobol) and [`get_GPEI`](/api/data.html#.data.users.adamobeng.fbsource.fbcode.ax.ax.modelbridge.factory.get_GPEI)), the following transforms will be applied by default, in this sequence:
* [`RemoveFixed`](../api/modelbridge.html#ax.modelbridge.transforms.remove_fixed.RemoveFixed): Remove [`FixedParameters`](../api/core.html#ax.core.parameter.FixedParameter) from the search space.
* [`OrderedChoiceEncode`](../api/modelbridge.html#ax.modelbridge.transforms.choice_encode.OrderedChoiceEncode): [`ChoiceParameters`](../api/core.html#ax.core.parameter.ChoiceParameter) with `is_ordered` set to `True` are encoded as a sequence of integers.
* [`OneHot`](../api/modelbridge.html#ax.modelbridge.transforms.one_hot.OneHot): [`ChoiceParameters`](../api/core.html#ax.core.parameter.ChoiceParameter) with `is_ordered` set to `False` are one-hot encoded.
* [`IntToFloat`](../api/modelbridge.html#ax.modelbridge.transforms.int_to_float.IntToFloat): Integer-valued [`RangeParameters`](../api/core.html#ax.core.parameter.RangeParameter) are converted to have float values.
* [`Log`](../api/modelbridge.html#ax.modelbridge.transforms.log.Log): [`RangeParameters`](../api/core.html#ax.core.parameter.RangeParameter) with `log_scale` set to `True` are log transformed.
* [`UnitX`](../api/modelbridge.html#ax.modelbridge.transforms.unit_x.UnitX): All float [`RangeParameters`](../api/core.html#ax.core.parameter.RangeParameter) are mapped to `[0, 1]`.
* [`Derelativize`](../api/modelbridge.html#ax.modelbridge.transforms.derelativize.Derelativize): Constraints relative to status quo are converted to constraints on raw values.
* [`StandardizeY`](../api/modelbridge.html#ax.modelbridge.transforms.standardize_y.StandardizeY): The Y values for each metric are standardized (subtract mean, divide by standard deviation).

Each transform defines both a forward and backwards transform. Arm parameters are passed through the forward transform before being sent along to the Model. The Model works entirely in the transformed space, and when new candidates are generated, they are passed through all of the backwards transforms so the [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) returns points in the original space.

New transforms can be implemented by creating a subclass of [`Transform`](../api/modelbridge.html#ax.modelbridge.transforms.base.Transform), which defines the interface for all transforms. There are separate methods for transforming the search space, optimization config, observation features, and observation data. Transforms that operate on only some aspects of the problem do not need to implement all methods, for instance, [`Log`](../api/modelbridge.html#ax.modelbridge.transforms.log.Log) implements only [`transform_observation_features`](../api/modelbridge.html#ax.modelbridge.transforms.log.Log.transform_observation_features) (to log transform the parameters), [`transform_search_space`](../api/modelbridge.html#ax.modelbridge.transforms.log.Log.transform_search_space) (to log transform the search space bounds), and [`untransform_observation_features`](../api/modelbridge.html#ax.modelbridge.transforms.log.Log.untransform_observation_features) (to apply the inverse transform).

The (ordered) list of transforms to apply is an input to the ModelBridge, and so can easily be altered to add new transforms. It is important that transforms be applied in the right order. For instance, the `StandardizeY` and `Winsorize` transforms both transform the observed metric values. Applying them in the order `[StandardizeY, Winsorize]` could produce very different results than `[Winsorize, StandardizeY]`. In the former case, outliers would have already been included in the standardization (a procedure sensitive to outliers), and so the second approach that winsorizes first is preferred.

See [the API reference](../api/modelbridge.html#transforms) for the full collection of implemented transforms.

## Implementing new models

The structure of the modeling stack makes it easy to implement new models and use them inside Ax. There are two ways this might be done.


### Using an existing Model interface

The easiest way to implement a new model is if it can be adapted to the one of the existing Model interfaces: ([`TorchModel`](api/models.html#ax.models.torch_base.TorchModel), [`DiscreteModel`](../api/models.html#ax.models.discrete_base.DiscreteModel), or [`RandomModel`](../api/models.html#ax.models.random.base.RandomModel)). The class definition provides the interface for each of the methods that should be implemented in order for Ax to be able to fully use the new model. Note however that not all methods must need be implemented to use some Ax functionality. For instance, an implementation of [`TorchModel`](../api/models.html#ax.models.torch_base.TorchModel) that implements only [`fit`](../api/models.html#ax.models.torch_base.TorchModel.fit) and [`predict`](../api/models.html#ax.models.torch_base.TorchModel.predict) can be used to fit data and make plots in Ax; however, it will not be able to generate new candidates (requires implementing [`gen`](../api/models.html#ax.models.torch_base.TorchModel.gen)) or be used with Ax's cross validation utility (requires implementing [`cross_validate`](../api/models.html#ax.models.torch_base.TorchModel.cross_validate)).

Once the new model has been implemented, it can be used in Ax with the corresponding [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) from the table above. For instance, suppose a new torch-based model was implemented as a subclass of [`TorchModel`](../api/models.html#ax.models.torch_base.TorchModel). We can use that model in Ax like:
```Python
new_model_obj = NewModel(init_args)  # An instance of the new model class
m = TorchModelBridge(
    experiment=experiment,
    search_space=search_space,
    data=data,
    model=new_model_obj,
    transforms=[UnitX, StandardizeY],  # Include the desired set of transforms
)
```
The [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) object `m` can then be used with plotting and cross validation utilities exactly the same way as the built-in models.

### Creating a new Model interface

If none of the existing Model interfaces work are suitable for the new model type, then a new interface will have to be created. This involves two steps: creating the new model interface and creating the new model bridge. The new model bridge must be a subclass of [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) that implements `ModelBridge._fit`,  `ModelBridge._predict`, `ModelBridge._gen`, and  `ModelBridge._cross_validate`. The implementation of each of these methods will transform the Ax objects in the inputs into objects required for the interface with the new model type. The model bridge will then call out to the new model interface to do the actual modeling work. All of the ModelBridge/Model pairs in the table above provide examples of how this interface can be defined. The main key is that the inputs on the [`ModelBridge`](../api/modelbridge.html#ax.modelbridge.base.ModelBridge) side are fixed, but those inputs can then be transformed in whatever way is desired for the downstream Model interface to be that which is most convenient for implementing the model.

```html
<script type="text/javascript" src="assets/slice.js"></script>
<script type="text/javascript" src="assets/contour.js"></script>
<script type="text/javascript" src="assets/cv.js"></script>
<script type="text/javascript" src="assets/fitted.js"></script>
```
