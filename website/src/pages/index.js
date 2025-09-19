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
import Layout from '@theme/Layout';

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
    <div className="col col--4 feature text--center">
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
};

const codeExample = `
>>> from ax import Client, RangeParameterConfig

>>> client = Client()
>>> client.configure_experiment(
      parameters=[
          RangeParameterConfig(
              name="x1",
              bounds=(-10.0, 10.0),
              parameter_type="float",
          ),
          RangeParameterConfig(
              name="x2",
              bounds=(-10.0, 10.0),
              parameter_type="float",
          ),
      ],
)
>>> client.configure_optimization(objective="-1 * booth")

>>> for _ in range(20):
>>>     for trial_index, parameters in client.get_next_trials(max_trials=1).items():
>>>         client.complete_trial(
>>>             trial_index=trial_index,
>>>             raw_data={
>>>                 "booth": (parameters["x1"] + 2 * parameters["x2"] - 7) ** 2
>>>                 + (2 * parameters["x1"] + parameters["x2"] - 5) ** 2
>>>             },
>>>         )

>>> client.get_best_parameterization()
{'x1': 1.02, 'x2': 2.97}  # true min is (1, 3)
`;

const QuickStart = () => (
  <div
    className="get-started-section padding--xl"
    style={{'background-color': 'var(--ifm-color-emphasis-200)'}}
    id="quickstart">
    <h2 className="text--center padding--md">Get Started</h2>
    <ol>
      <li>
        Install Ax:
        <Tabs>
          <TabItem value="linux" label="Linux" deafult>
            <CodeBlock
              languague="bash"
              showLineNumbers>{`pip3 install ax-platform`}</CodeBlock>
          </TabItem>
          <TabItem value="mac" label="Mac">
            <CodeBlock
              language="bash"
              showLineNumbers>{`conda install pytorch -c pytorch
pip3 install ax-platform`}</CodeBlock>
          </TabItem>
        </Tabs>
      </li>
      <li>
        Run an optimization:
        <br />
        <br />
        <CodeBlock language="python" showLineNumbers>
          {codeExample}
        </CodeBlock>
      </li>
    </ol>
  </div>
);


const papertitle = `Ax: A Platform for Adaptive Experimentation`;
const paper_bibtex = `
@inproceedings{olson2025ax,
  title = {{Ax: A Platform for Adaptive Experimentation}},
  author = {
    Olson, Miles and Santorella, Elizabeth and Tiao, Louis C. and
    Cakmak, Sait and Eriksson, David and Garrard, Mia and Daulton, Sam and
    Balandat, Maximilian and Bakshy, Eytan and Kashtelyan, Elena and
    Lin, Zhiyuan Jerry and Ament, Sebastian and Beckerman, Bernard and
    Onofrey, Eric and Igusti, Paschal and Lara, Cristian and
    Letham, Benjamin and Cardoso, Cesar and Shen, Shiyun Sunny and
    Lin, Andy Chenyuan and Grange, Matthew
  },
  booktitle = {AutoML 2025 ABCD Track},
  year = {2025}
}`;

const Reference = () => (
  <div
    className="padding--lg"
    id="reference"
    style={{}}>
    <h2 className='text--center'>References</h2>
    <div>
      <a href={`https://openreview.net/forum?id=U1f6wHtG1g`}>{papertitle}</a>
      <CodeBlock className='margin-vert--md'>{paper_bibtex}</CodeBlock>
    </div>
  </div>
);

const MyPage = () => {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <div className="hero text--center" style={{height: '40rem'}}>
        <div className="container">
          <div className="padding-vert--md">
            <img
              src={useBaseUrl('img/ax_wireframe.svg')}
              alt="Project Logo"
              style={{width: '300px'}}
            />
            <p className="hero__subtitle text--secondary">
              {siteConfig.tagline}
            </p>
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
              to="/docs/tutorials/quickstart/"
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
      <Reference />
    </Layout>
  );
};

export default MyPage;
