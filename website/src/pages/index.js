/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from '@theme/CodeBlock';
import Layout from "@theme/Layout";

const features = [
  {
    content:
      'Easy to plug in new algorithms and use the library across ' +
      'different domains.',
    image: 'img/th-large-solid.svg',
    title: 'Modular',
  },
  {
    content:
      'Field experiments require a range of considerations ' +
      'beyond standard optimization problems.',
    image: 'img/dice-solid.svg',
    title: 'Supports A/B Tests',
  },
  {
    content:
      'Support for industry-grade experimentation ' +
      'and optimization management, including MySQL storage.',
    image: 'img/database-solid.svg',
    title: 'Production-Ready',
  },
];

const Feature = ({imageUrl, title, content, image}) => {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className='col col--4 feature text--center'>
      {imgUrl && (
        <div>
          <img src={imgUrl} alt={title} />
        </div>
      )}
      {image && (
        <div>
          <img
            className="margin--md"
            src={image}
            alt={title}
            style={{width: '80px', height: '80px'}}
          />
        </div>
      )}
      <h2>{title}</h2>
      <p>{content}</p>
    </div>
  );
}

const codeExample = `>>> from ax import optimize
>>> best_parameters, best_values, experiment, model = optimize(
      parameters=[
        {
          "name": "x1",
          "type": "range",
          "bounds": [-10.0, 10.0],
        },
        {
          "name": "x2",
          "type": "range",
          "bounds": [-10.0, 10.0],
        },
      ],
      # Booth function
      evaluation_function=lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
      minimize=True,
  )

>>> best_parameters
{'x1': 1.02, 'x2': 2.97}  # true min is (1, 3)`;

const QuickStart = () => (
  <div
    className="get-started-section padding--xl"
    style={{'background-color': 'var(--ifm-color-emphasis-200)'}}
    id="quickstart"
  >
    <h2 className="text--center padding--md">Get Started</h2>
    <ol>
      <li>
        Install Ax:
        <Tabs>
          <TabItem value="linux" label="Linux" deafult>
              <CodeBlock languague="bash" showLineNumbers>{`pip3 install ax-platform`}</CodeBlock>
          </TabItem>
          <TabItem value="mac" label="Mac">
              <CodeBlock language="bash" showLineNumbers>{`conda install pytorch torchvision -c pytorch
pip3 install ax-platform`}</CodeBlock>
          </TabItem>
        </Tabs>
      </li>
      <li>
        Run an optimization:
        <br/><br/>
        <CodeBlock language="python" showLineNumbers>{codeExample}</CodeBlock>
      </li>
    </ol>
  </div>
);

const MyPage = () => {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <div className="hero text--center" style={{height: "40rem"}}>
        <div className="container">
          <div className="padding-vert--md">
            <img src={useBaseUrl('img/ax_wireframe.svg')} alt="Project Logo" style={{ width: "300px"}} />
            <p className="hero__subtitle text--secondary">{siteConfig.tagline}</p>
          </div>
          <div>
            <Link
              to="/docs/why-ax"
              className="button button--lg button--outline button--secondary margin--sm">
              Why Ax?
            </Link>
            <Link
              to="/docs/installation"
              className="button button--lg button--outline button--secondary margin--sm">
              Get started
            </Link>
            <Link
              to="/docs/tutorials/"
              className="button button--lg button--outline button--secondary margin--sm">
              Tutorials
            </Link>
          </div>
        </div>
      </div>
      <div className="padding--xl">
        <h2 className="text--center padding--md">Key Features</h2>
        {features && features.length > 0 && (
          <div className="row">
            {features.map(({title, imageUrl, content, image}) => (
              <Feature
                key={title}
                title={title}
                imageUrl={imageUrl}
                content={content}
                image={image}
              />
            ))}
          </div>
        )}
      </div>
      <QuickStart />
    </Layout>
  );
};

export default MyPage;
