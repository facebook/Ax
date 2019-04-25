---
id: banditopt
title: Bandit Optimization
---

[Bayesian optimization](bayesopt.md) provides a solution for parameter tuning in problems with continuous parameters. 
However, many decision problems simply require choosing from among a set of discrete candidates, rather than finding points in a continuous space.  Most ordinary A/B tests fall into this category—there are a handful of choices to evaluate against each other. A fixed percentage of traffic goes to each of these choices, and after a few days/weeks of splitting traffic, a winner is chosen. With more than a few choices, however, A/B tests quickly become prohibitively resource-intensive. With a large number of choices, each candidate receives a small amount of traffic, even if it appears to be the best choice.

Bandit optimization aims to provide a smarter way of evaluating the performance of these discrete choices. Ordinary A/B testing sets a fixed split of traffic among candidates. Bandit optimization, on the other hand, sequentially updates the allocation of traffic to each candidate, based on its performance so far. The key problem for these algorithms is balancing exploration—sending traffic to candidates that have the potential to perform well—with exploitation, sending traffic to candidates that already appear to perform well. This trade-off is very similar to the underlying exploration problem highlighted in Bayesian Optimization [acquisition functions](bayesopt.md#acquisition-functions).  However, while modeling the relationships between parameters in the search space is a necessity in Bayesian optimization, bandit optimization typically treats each arm as independent. This makes bandit optimization more sample efficient and simpler to analyze on discrete-choice problems. 

Bandit optimization can be more sample efficient than traditional A/B tests, where the number of samples by treatment group is set beforehand and is generally balanced. Consequently, it is safer with larger cohorts because the samples are automatically diverted towards the good parameter values. They are also suitable for test beds in production applications that are used for continuous evaluation of newly introduced features.


## How does it work?

Ax relies on a simple and effective algorithm for performing bandit optimization: Thompson sampling. This method has a clear intuition: select a parameter configuration with a probability proportional to that value being the best. This algorithm is simple to implement and has strong guarantees of converging to a set of arms or parameter configuration that is close to the best — all without any human intervention. To understand how this works, we describe an advertising optimization problem where we want to choose parameter configurations that maximize the click-through (CTR) or conversion rate and the rewards are binary - either clicks (successes) or views without clicks (failures).

Let's assume that the probability for each arm to be successful has a beta prior distribution. Given a set of clicks and views without clicks for any arm, a Bayesian update leads to a beta posterior distribution and an updated estimate of the probability of selecting that arm. In Thompson sampling, a sample is drawn from the posterior distribution for each arm and the arm with the largest sampled value is selected for the next evaluation. This process is repeated by updating the selection probabilities based on observed CTRs in predefined time windows and over many iterations, the algorithm converges on a few arms that lead to better CTRs.

The following figure is a simulated example of how the assignment probabilities for an experiment with 10 arms or experimental conditions may evolve over 20 iterations:

![Bandit Optimization Allocations](assets/mab_probs.png)

Starting with equal assignment probabilities for each arm, every round of bandit optimization produces an updated set of assignment probabilities (represented here by the height of the colored bars in each column) based on the average CTR from 10 users per round. Since the true CTR is highest for the second arm followed by the first arm in this simulated example, the assignment probabilities of these arms have clearly dominate the other arms over 20 rounds of bandit optimization.  

The spread of the posterior distribution depicts the uncertainty around the true CTR for each arm and allows the bandit optimization algorithm to quickly sample or explore the arms in the initial rounds. This causes the uncertainty in all the arms to drop sufficiently to pave the way for exploitation. The following figure animates the average observed CTRs (blue x), the assignment probabilities (solid round symbol) and the uncertainties (gray error bars) based on the posterior distributions after each round of experimentation for the previous example. The arms 3 through 8 are sampled just enough to get a rough estimate of the low CTRs, before converging on the first two arms with high CTRs. This example can be viewed as a discretized version of the animated example of [Bayesian optimization](bayesopt.md).

![Bandit Optimization: Posteriors](assets/mab_animate.gif)


## Regret

We want a bandit algorithm to maximize the total rewards over time or alternatively, minimize the regret, which is defined as the cumulative difference between the highest possible reward and the actual reward at point in time. A smaller regret is a measure of how well the algorithm is able to balance the exploration vs. exploitation trade-off - too much of either exploration (as in factorial experiments), or exploitation (as in purely greedy algorithms) increases the regret. We ideally want the regret to increase slowly with successive rounds of experimentation. In this sense, it can be used as a performance metric to evaluate different bandit algorithms.

The following figure compares the average regret of three different approaches to bandit optimization for a 10 arm bandit problem over 200 rounds of experimentation:

1. Thompson sampling 
2. Greedy: select the arm with the current best reward
3. Eepsilon-greedy: approach that either picks an arm randomly with probability e or proceeds greedily with probability 1 - e. Setting e = 0 leads to a the purely greedy approach and setting e = 1 leads to a purely exploratory approach.

![Bandit Optimization: Regret](assets/mab_regret.png)

The regret of the purely greedy approach is the highest amongst the three approaches and a little bit of exploration as in the epsilon-greedy approach with e = 0.1 leads to a much smaller regret over time. Thompson sampling balances the tradeoff between exploration and exploitation very well, and out-performs the other two approaches.

