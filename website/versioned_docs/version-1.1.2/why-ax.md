---
id: why-ax
title: Why Ax?
---

# Why Ax?

Developers and researchers alike face problems which confront them with a large
space of possible ways to configure something –– whether those are learning
rates or other hyperparameters in machine learning, "magic numbers" used for
infrastructure or compiler flags, or design parameters in a physical engineering
task. Selecting and tuning these configurations can often take time, resources,
and affects the quality of user experiences. Ax is a machine learning system to
help guide and automate this experimentation process, so that researchers and
developers can determine how to get the most out of their processes in an
efficient manor.

Ax is a platform for optimizing many kinds of experiment, and is typically
useful for problems that are expensive to evaulate or where the number of
evaluations must remain limited. Machine learning experiments, A/B tests, and
costly simulations are contexts in which adaptive experimentation techniques are
especially useful. Ax can optimize continuous (e.g., integer or floating
point)-valued configurations, discrete configurations (e.g., variants of an A/B
test), or mixed spaces using techniques like
[Bayesian optimization](./intro-to-bo.mdx). This makes it suitable for a wide range
of applications.

# Unique capabilities

- **Expressive API**: Ax has an expressive API that can address many real-world
  optimization tasks. It handles complex search spaces, multiple objectives,
  constraints on both parameters and outcomes, and noisy observations. It
  supports suggesting multiple designs to evaluate in parallel (both
  synchronously and asynchronously) and the ability to early-stop evaluations.

- **Strong performance out of the box**: Ax abstracts away optimization details
  that are important but obscure, providing sensible defaults and enabling
  practitioners to leverage advanced techniques otherwise only accessible to
  optimization experts.

- **State-of-the-art methods**: Ax leverages state-of-the-art
  [Bayesian optimization](./intro-to-bo.mdx) algorithms implemented in
  [BoTorch](https://botorch.org/), to deliver strong performance across a
  variety of problem classes.

- **Flexible:** Ax is highly configurable, allowing researchers to plug in novel
  optimization algorithms, models, and experimentation flows.

- **Production ready:** Ax offers automation and orchestration features as well
  as robust error handling for real-world deployment at scale.
