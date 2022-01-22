---
id: why-ax
title: Why Ax?
sidebar_label: Why Ax?
---

Developers and researchers alike face problems which confront them with a large space of possible ways to configure something –– whether those are "magic numbers" used for infrastructure or compiler flags, learning rates or other hyperparameters in machine learning, or images and calls-to-action used in marketing promotions.  Selecting and tuning these configurations can often take time, resources, and can affect the quality of user experiences.  Ax is a machine learning system to help automate this process, so that researchers and developers can determine how to get the most out of their software in an optimally efficient way.

Ax is a platform for optimizing any kind of experiment, including machine learning experiments, A/B tests, and simulations.  Ax can optimize discrete configurations (e.g., variants of an A/B test) using multi-armed bandit optimization, and continuous (e.g., integer or floating point)-valued configurations using Bayesian optimization. This makes it suitable for a wide range of applications.

Ax has been successfully applied to a variety of product, infrastructure, ML, and research applications at Facebook.

# Unique capabilities
- **Support for noisy functions**.  Results of A/B tests and simulations with reinforcement learning agents often exhibit high amounts of noise.  Ax supports [state-of-the-art algorithms](https://research.facebook.com/blog/2018/09/efficient-tuning-of-online-systems-using-bayesian-optimization/) which work better than traditional Bayesian optimization in high-noise settings.
- **Customization**.  Ax's developer API makes it easy to integrate custom data modeling and decision algorithms. This allows developers to build their own custom optimization services with minimal overhead.
- **Multi-modal experimentation**.  Ax has first-class support for running and combining data from different types of experiments, such as "offline" simulation data and "online" data from real-world experiments.
- **Multi-objective optimization**. Ax supports multi-objective and constrained optimization which are common to real-world problems, like improving load time without increasing data use.
