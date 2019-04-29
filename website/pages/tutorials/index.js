/**
 * Copyright (c) 2019-present, Facebook, Inc.
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
    const {baseUrl} = this.props;

    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">Welcome to Ax Tutorials</h1>
            </header>
              <p>
                Here you can learn about the structure and applications of Ax
                from examples.
              </p>
              <p>
                <a href="building_blocks.html">Building Block of Ax</a>&nbsp;
                is a good place to start, as it teaches about the architecture
                of Ax and explains the experimentation/optimization process.
              </p>
              <p>
                Furtner, our Bayesian Optimization tutorials include:
              </p>
              <ul>
                <li>
                  <a href="tune_cnn.html">HPO for PyTorch</a>
                  &nbsp; exemplifies hyperparameter optimization with Ax and integration
                  with an external ML library.
                </li>
              </ul>
              <ul>
                <li>
                  <a href="multi_task.html">Multi-Task Modeling</a>
                  &nbsp; illustrates multi-task Bayesian Optimization on a constrained
                  synthetic Hartmann6 problem.
                </li>
              </ul>
              <ul>
                <li>
                  <a href="benchmarking_suite_example.html">Benchmarking Suite</a>
                  &nbsp; demonstrates how to use the Ax benchmarking suite to
                  compare Bayesian Optimization algorithm performances and
                  generate a comparative report with visualizations.
                </li>
              </ul>
              <p>
                For experiments done in a real-life setting, refer to our field
                experiments tutorials:
              </p>
              <ul>
                <li>
                  <a href="factorial.html">Bandit Optimization</a>
                  &nbsp; shows how Thompson Sampling can be used to intelligently
                  reallocate resources to well-performing configurations in real-time.
                </li>
              </ul>
              <ul>
                <li>
                  <a href="human_loop.html">Human-in-the-Loop Optimization</a>
                  &nbsp; walks through manually influencing the course of
                  optimization in real-time.
                </li>
              </ul>
              <p>
                Finally, each of our 3 API tutorials:&nbsp;
                <a href="gpei_hartmann_loop.html">Loop</a>,&nbsp;
                <a href="gpei_hartmann_service.html">Service</a>, and&nbsp;
                <a href="gpei_hartmann_developer.html">Developer</a> –– showcases
                optimization on a constrained Hartmann6 problem, with each
                respective API.
              </p>
           </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
