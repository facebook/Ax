/**
 * Copyright (c) 2019-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
        <TutorialSidebar currentTutorialID={null}/>
          <Container className="mainContainer documentContainer postContainer">
                <div className="post">
                  <header className="postHeader">
                    <h1 className="postHeaderTitle">Welcome to Ax Tutorials</h1>
                  </header>
	    <body>
	    <p>Ax is a machine learning system designed to help automate the
	    process of selecting and tuning ML configurations. 
            </p><p>
	    The tutorials here will help you understand how to use
	    Ax to tune and optimize your neural networks.
	    </p>
	    <ul>
	    <li><a href="tune_cnn.html">Tune a CNN on MNIST</a></li>
            <li><a href="multi_task.html">Mixed online/offline Bayesian Optimization in Ax</a></li>
            <li><a href="benchmarking_suite_example.html">The Ax Benchmarking Suite</a></li>
	    <li><a href="factorial.html">Factorial design with Empirical Bayes and Thompson Sampling</a></li>
	    <li><a href="managed_loop.html">Ax Managed Loop Example</a></li>
	    <li><a href="service_api.html">Ax Service-like API</a></li>
	    <li><a href="tune_cnn.html">Tune a CNN on MNIST</a></li>
	    </ul>
	    </body>
                </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
