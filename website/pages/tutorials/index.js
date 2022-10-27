/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">Welcome to Ax Tutorials</h1>
            </header>
            <p>
              Here you can learn about the structure and applications of Ax from
              examples.
            </p>
            <p>
              <b>Our 3 API tutorials:</b>&nbsp;
              <a href="gpei_hartmann_loop.html">Loop</a>,&nbsp;
              <a href="gpei_hartmann_service.html">Service</a>, and&nbsp;
              <a href="gpei_hartmann_developer.html">Developer</a> &mdash; are a
              good place to start. Each tutorial showcases optimization on a
              constrained Hartmann6 problem, with the Loop API being the
              simplest to use and the Developer API being the most customizable.
            </p>
            <p>
              <b>
                Further, we explore the different components available in Ax in
                more detail.
              </b>{' '}
              The components explored below serve to set up an experiment,
              visualize its results, configure an optimization algorithm, run an
              entire experiment in a managed closed loop, and combine BoTorch
              components in Ax in a modular way.
            </p>
            <ul>
              <li>
                <a href="visualizations.html">Visualizations</a>&nbsp;
                illustrates the different plots available to view and understand
                your results.
              </li>
            </ul>
            <ul>
              <li>
                <a href="generation_strategy.html">GenerationStrategy</a>&nbsp;
                steps through setting up a way to specify the optimization
                algorithm (or multiple). A <code>GenerationStrategy</code>
                is an important component of Service API and the
                <code>Scheduler</code>.
              </li>
            </ul>
            <ul>
              <li>
                <a href="scheduler.html">Scheduler</a>&nbsp; demonstrates an
                example of a managed and configurable closed-loop optimization,
                conducted in an asyncronous fashion. <code>Scheduler</code> is a
                manager abstraction in Ax that deploys trials, polls them, and
                uses their results to produce more trials.
              </li>
            </ul>
            <ul>
              <li>
                <a href="modular_botax.html">
                  Modular <code>BoTorchModel</code>
                </a>
                &nbsp; walks though a new beta-feature &mdash; an improved
                interface between Ax and <a href="https://botorch.org/">BoTorch</a>{' '}
                &mdash; which allows for combining arbitrary BoTorch components
                like
                <code>AcquisitionFunction</code>, <code>Model</code>,
                <code>AcquisitionObjective</code> etc. into a single{' '}
                <code>Model</code> in Ax.
              </li>
            </ul>
            <p>
              <b>Our other Bayesian Optimization tutorials include:</b>
            </p>
            <ul>
              <li>
                <a href="tune_cnn.html">
                  Hyperparameter Optimization for PyTorch
                </a>
                &nbsp; provides an example of hyperparameter optimization with
                Ax and integration with an external ML library.
              </li>
            </ul>
            <ul>
              <li>
                <a href="raytune_pytorch_cnn.html">
                  Hyperparameter Optimization via Raytune
                </a>
                &nbsp; provides an example of parallelized hyperparameter
                optimization using Ax + Raytune.
              </li>
            </ul>
            <ul>
              <li>
                <a href="multi_task.html">Multi-Task Modeling</a>
                &nbsp; illustrates multi-task Bayesian Optimization on a
                constrained synthetic Hartmann6 problem.
              </li>
            </ul>
            <ul>
              <li>
                <a href="multiobjective_optimization.html">
                  Multi-Objective Optimization
                </a>
                &nbsp; demonstrates Multi-Objective Bayesian Optimization on a
                synthetic Branin-Currin test function.
              </li>
            </ul>
            <ul>
              <li>
                <a href="early_stopping.html">
                  Trial-Level Early Stopping
                </a>
                &nbsp; shows how to use trial-level early stopping on an ML
                training job to save resources and iterate faster.
              </li>
            </ul>

            {/* <ul>
              <li>
                <a href="benchmarking_suite_example.html">Benchmarking Suite</a>
                &nbsp; demonstrates how to use the Ax benchmarking suite to
                compare Bayesian Optimization algorithm performances and
                generate a comparative report with visualizations.
              </li>
            </ul> */}
            <p>
              For experiments done in a real-life setting, refer to our field
              experiments tutorials:
            </p>
            <ul>
              <li>
                <a href="factorial.html">Bandit Optimization</a>
                &nbsp; shows how Thompson Sampling can be used to intelligently
                reallocate resources to well-performing configurations in
                real-time.
              </li>
            </ul>
            <ul>
              <li>
                <a href="human_in_the_loop/human_in_the_loop.html">
                  Human-in-the-Loop Optimization
                </a>
                &nbsp; walks through manually influencing the course of
                optimization in real-time.
              </li>
            </ul>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
