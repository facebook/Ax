---
id: data
title: Data
---

## Fetching Data

[Metrics](glossary.md#metric) provide an interface for fetching data for an experiment or trial. Experiment objectives and outcome constraints are special types of metrics, and you can also attach additional metrics for tracking purposes.

Each metric is responsible for fetching its own data. Thus, all metric classes must implement the method `fetch_trial_data`, which accepts a [`Trial`](../api/core.html#ax.core.trial.Trial) and returns an instance of [`Data`](../api/core.html#ax.core.data.Data), a wrapper around a Pandas DataFrame.

To fetch data for an experiment or trial, use `exp.fetch_data` or `trial.fetch_data`. These methods fetch data for all metrics on the experiment and then combine the results into a new aggregate [`Data`](../api/core.html#ax.core.data.Data) instance.

Each row of the final DataFrame represents the evaluation of an arm on a metric. As such, the required columns are: `arm_name`, `metric_name`, `mean`, and `sem`. Additional optional columns are also supported: `trial_index`, `start_time`, and `end_time`.

| arm_name | metric_name | mean | sem |
|----------|-------------|------|-----|
| 0_0      | metric1     | ...  | ... |
| 0_0      | metric2     | ...  | ... |
| 0_1      | metric1     | ...  | ... |
| 0_1      | metric2     | ...  | ... |

## Adding Your Own Metric

Our base Metric class is meant to be subclassed. Subclasses must provide an implementation of `fetch_trial_data`.

An example of a custom metric:

```python
import pandas as pd
from ax import Metric

class CustomMetric(Metric):

    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": 0.0,  # mean value of this metric when this arm is used
                "sem": 0.0,  # standard error of the above mean
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))
```

## Advanced Data Fetching

If you need to fetch data for multiple metrics or trials simultaneously,
your Metric can implement the methods `fetch_experiment_data`, `fetch_trial_data_multi`,
and `fetch_experiment_data_multi`. The default implementations of these methods
use `fetch_trial_data` internally, but can be overridden if bulk data fetching
is more appropriate for the metric type.
