---
id: why-ax
title: Why Ax?
sidebar_label: Why Ax?
---

Developers and researchers alike face problems where they are confronted with an a large space of possible ways to configure something---whether that be "magic numbers" used for infrastructure or compiler flags, learning rates or neural net widths in machine learning, or images and calls to action used in marketing promotions.  Selecting and tuning these configurations can often take time, resources, and could affect user experiences.  We developed Ax, a machine learning system to help automate this process, and help researchers and developers get the most out of their software in an optimally efficient way.

Ax is a platform for optimizing any kind of experiment, including machine learning experiments, A/B tests, and simulations.  Ax includes for optimizing both discrete configurations (e.g., variants of an A/B test) using multi-armed bandit optimization and continuous (e.g., integer or floating point)-valued configurations using Bayesian optimization suitable for a wide range of applications.  Ax has been successfully applied to a variety of product, infrastructure, ML, and research applications at Facebook.

# Unique capabilities

- **Support for noisy functions.**.  Results of A/B tests and simulations with reinforcement learning agents often exhibit high amounts of noise.  Ax supports [state-of-the-art algorithms](https://research.fb.com/efficient-tuning-of-online-systems-using-bayesian-optimization/) that work better than traditional Bayesian optimization in high-noise settings.
- **Customizability**.  Ax's developer API enables custom data modeling and decision algorithms that make Ax broadly applicable to emerging applications, and allows developers to build their own, custom optimization services.
- **Multi-modal experimentation**.  Ax has first-class support for running and combining data from different types of experiments, such as "offline" simulation data and "online" data from real-world experiments.
- **Multi-objective optimization**. Ax supports multi-objective and constrained optimization, which are common to real-world problems.  Examples of constrained optimization include "improve load time without increasing data use".

# APIs

Ax runs locally on your own computer or server, and can be used through one of three different APIs.

- Developer API. This API is for ad hoc use by data scientists, machine learning engineers, and researchers.  The developer API allows for a great deal of customizability, and is recommended for those who plan to use Ax to optimize A/B tests.
- Service API. This is a simplified RESTful-style API which can be used as a lightweight service (via, for example, a REST or Thirft interface) for simple parameter tuning applications, like AutoML and simulation optimization.
- Managed API. This API is intended for large-scale, custom custom production services where data may arrive from different sources (such as databases) in an asynchronous way.

In addition, a command-line interface is available.