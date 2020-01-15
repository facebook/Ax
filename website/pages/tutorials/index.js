/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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
              Our 3 API tutorials:&nbsp;
              <a href="gpei_hartmann_loop.html">Loop</a>,&nbsp;
              <a href="gpei_hartmann_service.html">Service</a>, and&nbsp;
              <a href="gpei_hartmann_developer.html">Developer</a> &mdash; are a
              good place to start. Each tutorial showcases optimization on a
              constrained Hartmann6 problem, with the Loop API being the
              simplest to use and the Developer API being the most customizable.
            </p>
            <p>Further, our Bayesian Optimization tutorials include:</p>
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
                &nbsp; provides an example of hyperparameter optimization with
                Ax + Raytune.
              </li>
            </ul>
            <ul>
              <li>
                <a href="multi_task.html">Multi-Task Modeling</a>
                &nbsp; illustrates multi-task Bayesian Optimization on a
                constrained synthetic Hartmann6 problem.
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
            <p>
              Finally, we explore the different components available in Ax in
              more detail, both for setting up the experiment and visualizing
              results.
            </p>
            <ul>
              <a href="building_blocks.html">Building Blocks of Ax</a>&nbsp;
              examines the architecture of Ax and the
              experimentation/optimization process.
            </ul>
            <ul>
              <a href="visualizations.html">Visualizations</a>&nbsp; illustrates
              the different plots available to view and understand your results.
            </ul>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
