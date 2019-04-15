---
id: data
title: Data
---

## Fetching Data

Metrics provide an interface for fetching data for an experiment or trial. Experiment objectives and outcome constraints are special types of metrics, and you can also attach additional metrics for tracking purposes.

Our base Metric class is meant to be subclassed, as you will need to provide a custom implementation of ```fetch_trial_data```. This method accepts a Trial and returns an instance of Data, which is a wrapper around a Pandas dataframe. Additional methods for fetching data for multiple trials or an entire experiment are provided with default implementations that use ```fetch_trial_data internally```, but can also be subclassed.

Each row of the dataframe represents the evaluation of an arm on a metric. As such, the required columns are  arm_name, metric_name, mean, and sem. Additional optional columns are also supported: trial_index, start_time, and end_time.

## Adding Your Own Metric

To add a custom metric, subclass ```Metric``` and implement ```fetch_trial_data```:

```python
class CustomMetric(Metric):

    def fetch_trial_data(self, trial: BaseTrial, kwargs: Dict[str, Any]) → Data:    
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": # mean value of this metric when this arm is used
                "sem": # standard error of the above mean
                "trial_index": trial.index,
            })
        )
        return Data(df=pd.DataFrame.from_records(records))
```
