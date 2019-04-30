---
id: banditopt
title: Bandit Optimization
---

Many decision problems require choosing from among a discrete set of candidates; for these problems we turn to bandit optimization. [Bayesian optimization](bayesopt.md), on the other hand, provides a solution for parameter tuning in problems with continuous parameters. Most ordinary A/B tests fall into the category of bandit optimization. There are typically a handful of choices to evaluate against each other, a fixed percentage of traffic goes to each choice, and after a few days/weeks of splitting traffic, a winner is chosen. With more than a few choices, however, A/B tests quickly become prohibitively resource-intensive, largely because all choices -- no matter how good or bad they appear -- receive the same traffic allocation.

Bandit optimization aims to provide a smarter way of allocating units among these discrete choices by sequentially updating the allocation of traffic to each candidate, based on its performance so far. The key problem for these algorithms is balancing exploration—sending traffic to candidates that have the potential to perform well—with exploitation, sending traffic to candidates that already appear to perform well. This trade-off is very similar to the underlying exploration problem highlighted in Bayesian Optimization [acquisition functions](bayesopt.md#acquisition-functions).

Bandit optimization is more sample efficient than traditional static A/B tests. Consequently, it is safer with larger cohorts because the samples are automatically diverted towards the good parameter values (and away from the bad ones).


## How does it work?

Ax relies on a simple and effective algorithm for performing bandit optimization: [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling). This method has a clear intuition: select a parameter configuration with a probability proportional to that value being the best. This algorithm is simple to implement and has strong guarantees of converging to a set of arms or parameter configuration that is close to the best — all without any human intervention. To understand how this works, we describe an advertising optimization problem where we want to choose parameter configurations that maximize the click-through (CTR) or conversion rate and the rewards are binary - either clicks (successes) or views without clicks (failures).

As we run the experiment, we develop more precise esimates of the performance of each arm. In Thompson sampling, we draw samples from the distribution of plausible effects for each arm and the largest sampled value is recorded. We repeat this process many times and the resulting distribution of maximal arms is how we assign users to arms in the future. This rapidly winnows down arms to only the very best.

The following figure is an example of how assignment probabilities for an experiment with 10 arms may evolve over 20 iterations of batch-based Thompson sampling:

![Bandit Optimization Allocations](assets/mab_probs.png)

The process starts by assigning all arms equally likely. Bandit optimizaiton then produces updated assignment probabilities (represented here by the height of the colored bars in each column) based on the average CTR observed up until that point. Since the true CTR is highest for the second arm followed by the first arm in this simulated example, those arms are given subsequently larger allocations over 20 rounds of bandit optimization.

Early in the process, the uncertainty in our estimates of CTR means that the bandit optimization spreads samples around among a diversity of arms. This, in turn, helps us better estimate *all* of the arms and start focusing in on the arms which perform well. The following figure animates how estimates evolve under bandit optimization. The small blue x indicates the observed CTRs within each round, while the solid round symbol (and gray error bars) indicate our aggregated estimates across all rounds. Arms 3 through 8 are sampled just often enough to get a rough estimate that their CTRs are low, as the algorithm focuses exploration on the first two arms to better identify which is the best. This example can be viewed as a discretized version of the animated example of [Bayesian optimization](bayesopt.md).

![Bandit Optimization: Posteriors](assets/mab_animate.gif)

## How well does it work?

We want a bandit algorithm to maximize the total rewards over time or equivalently, minimize the regret, which is defined as the cumulative difference between the highest possible reward and the actual reward at a point in time. In our running example, regret is the number of clicks we "left on the table" through our choice of allocation procedure. We can imagine two extremes:

1. Pure exploration, in which we just always allocate users evenly across all conditions. This is the standard approach to A/B tests.
2. Pure exploitation, in which we simply allocate all users to the arm we think is most likely to be best.

Both of these extremes will do a poor job of minimizing our regret, so our aim is to try and smartly balance them.

The following figure compares the cumulative regret of three different approaches to bandit optimization for 200 rounds of experimentation on our running example:

1. Thompson sampling: the primary approach used by Ax, described above
2. Greedy: select the arm with the current best reward
3. Epsilon-greedy: Randomly picks an arm e percent of the time, picks the current best arm 100-e% of the time.

![Bandit Optimization: Regret](assets/mab_regret.png)

The regret of the purely greedy approach is the highest amongst the three approaches. A little bit of exploration as in the epsilon-greedy approach with e = 10 leads to much smaller regret over time. Thompson sampling balances the tradeoff between exploration and exploitation very well, and out-performs the other two approaches.

It turns out, we can do even better than this by applying a simple model.

## Empirical Bayes

In short, our empirical Bayes model consists of taking noisy estimates from a bunch of arms and "shrinking" the outlying ones a bit towards the overall central tendency across all arms.

The specific method we use is [James-Stein estimation](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator). This method is linear, which means that if multiple arms have estimates with similar levels of precision, they will be moved towards the middle of the effect distribution proportionally to their distance from the middle. Doing this turns out to be optimal in the case of a Gaussian distribution of effects, but will improve accuracy even if that isn't the case (so long as there are [at least three means](https://projecteuclid.org/download/pdf_1/euclid.bsmsp/1200501656)).

Below are two experiments and how their estimates change as a result of applying the empirical Bayes estimator.

![Shrinkage in two representative experiments](assets/example_shrinkage.png)

The experiment on the left had large effects relative to estimation variability and so shrinkage (visualized here as distance from the dashed $y=x$ line) was very small. On the right side, however, we can see an experiment where shrinkage makes a real difference. Effects far from the center of the distribution result in fairly substantial shrinkage, reducing the range of effects by nearly half. While effect estimates in the middle were largely unchanged, the largest observed effects went from around 17% before shrinkage to around 8% afterwards.

The vast majority of experimental groups are estimated more accurately using empirical Bayes. The arms which tend to have increases in error are those with the largest effects. Understating the effects of such arms is usually not a very big deal when making launch decisions, however, as one is usually most interested in *which* arm is the best rather than exactly how good it is.

Empirical Bayes does a better job of playing the best arm than does using the raw effect estimates. It does this by concentrating exploration early in the experiment. In particular, it concentrates that exploration on the *set* of arms that look good, rather than over-exploiting the single best performing arm. By spreading exploration out a little bit more when effect estimates are noisy (and playing the best arm a little less), it is able to identify the best arm with more confidence later in the experiment.

We have a lot more of the [details in our paper](https://ddimmery.com/publication/experiment-shrinkage/).
