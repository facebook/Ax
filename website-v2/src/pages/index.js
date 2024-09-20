/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

const CompLibrary = {
  Container: props => <div {...props}>{props.children}</div>,
  GridBlock: props => <div {...props}></div>,
  MarkdownBlock: props => <div {...props}>{props.children}</div>
};

import CodeBlock from '@theme/CodeBlock';

import Layout from "@theme/Layout";

const MarkdownBlock = CompLibrary.MarkdownBlock;
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="splashLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div>{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={baseUrl + 'img/ax_wireframe.svg'} />
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href={docUrl('why-ax.html')}>Why Ax?</Button>
            <Button href={'#quickstart'}>Get Started</Button>
            <Button href={`${baseUrl}tutorials/`}>Tutorials</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
          children={props.children}
        />
      </Container>
    );

    const Description = () => (
      <Block background="light">
        {[
          {
            content:
              'This is another description of how this project is useful',
            image: `${baseUrl}img/docusaurus.svg`,
            imageAlign: 'right',
            title: 'Description',
          },
        ]}
      </Block>
    );

    const codeExample = `
>>> from ax import optimize
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
{'x1': 1.02, 'x2': 2.97}  # true min is (1, 3)
    `;

    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
      >
        <h2 style={{textAlign: 'center'}}>Get Started</h2>
        <Container>
          <ol>
            <li>
              Install Ax:
              <CodeBlock language="bash">{`conda install pytorch torchvision -c pytorch  # OSX only`}</CodeBlock>
              <CodeBlock languague="bash">{`pip3 install ax-platform  # all systems`}</CodeBlock>
            </li>
            <li>
              Run an optimization:
              <CodeBlock language="python">{codeExample}</CodeBlock>
            </li>
          </ol>
        </Container>
      </div>
    );

    const Features = () => (
      <div className="productShowcaseSection">
        <h2 style={{textAlign: 'center'}}>Key Features</h2>
        <Block layout="fourColumn">
          {[
            {
              content:
                'Easy to plug in new algorithms and use the library across ' +
                'different domains.',
              image: `${baseUrl}img/th-large-solid.svg`,
              imageAlign: 'top',
              title: 'Modular',
            },
            {
              content:
                'Field experiments require a range of considerations ' +
                'beyond standard optimization problems.',
              image: `${baseUrl}img/dice-solid.svg`,
              imageAlign: 'top',
              title: 'Supports A/B Tests',
            },
            {
              content:
                'Support for industry-grade experimentation ' +
                'and optimization management, including MySQL storage.',
              image: `${baseUrl}img/database-solid.svg`,
              imageAlign: 'top',
              title: 'Production-Ready',
            },
          ]}
        </Block>
      </div>
    );

    const features = [
      {
        content:
          'Easy to plug in new algorithms and use the library across ' +
          'different domains.',
        image: `${baseUrl}img/th-large-solid.svg`,
        imageAlign: 'top',
        title: 'Modular',
      },
      {
        content:
          'Field experiments require a range of considerations ' +
          'beyond standard optimization problems.',
        image: `${baseUrl}img/dice-solid.svg`,
        imageAlign: 'top',
        title: 'Supports A/B Tests',
      },
      {
        content:
          'Support for industry-grade experimentation ' +
          'and optimization management, including MySQL storage.',
        image: `${baseUrl}img/database-solid.svg`,
        imageAlign: 'top',
        title: 'Production-Ready',
      },
    ];

    const featuresNode = features && features.length && (
      <section className="productShowcaseSection">
        <div className="container">
        <h2 style={{textAlign: 'center'}}>Key Features</h2>
          <div className="row">
            {features.map(({image, title, content}, idx) => (
              <div
                key={idx}
                className="col col--4">
                {image && (
                  <div className="text--center">
                    <img
                      src={image}
                      alt={title}
                    />
                  </div>
                )}
                <h3>{title}</h3>
                <p>{content}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    )

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="landingPage mainContainer">
          {/* <Features /> */}
          {featuresNode}
          <QuickStart />
        </div>
      </div>
    );
  }
}

// export default props => <Layout><Index {...props} /></Layout>;


import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

const MyPage = () => {
  const {siteConfig} = useDocusaurusContext();

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
      className="get-started-section"
      id="quickstart"
    >
      <h2 style={{textAlign: 'center'}}>Get Started</h2>
      <Container>
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
      </Container>
    </div>
  );

  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <div className="hero text--center">
        <div className="container ">
          <div className="padding-vert--md">
            <h1 className="hero__title">{siteConfig.title}</h1>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
          </div>
          <div>
            <Link
              to="/docs/why-ax"
              className="button button--lg button--outline button--primary">
              Why Ax?
            </Link>
            <Link
              to="/docs/why-ax"
              className="button button--lg button--outline button--primary">
              Get started
            </Link>
            <Link
              to="/docs/why-ax"
              className="button button--lg button--outline button--primary">
              Tutorials
            </Link>
          </div>
        </div>
      </div>
      <QuickStart />
    </Layout>
  );
};

export default MyPage;
