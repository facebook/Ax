/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

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
      <div className="projectLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        {siteConfig.title}
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
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
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href={docUrl('why-ax.html')}>Why Ax?</Button>
            <Button href={"#quickstart"}>Get Started</Button>
            <Button href={docUrl('applications.html')}>Applications</Button>
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
    //getStartedSection

    const pre = "```";

    const codeExample = `${pre}python
>>> from ax.service.managed_loop import OptimizationLoop
>>> loop = OptimizationLoop.with_evaluation_function(
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
        experiment_name="test",
        objective_name="objective",
        evaluation_function=lambda p, _: {
          "objective": (
            (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5),  # mean (Booth function)
            0.0  # sem
          )
        },
        minimize=True,
    )

>>> loop.full_run()

>>> loop.get_best_point()
{'x1': 1.02, 'x2': 2.97}  # global min is (1, 3)
    `;

    const QuickStart = () => (
      <div
        className="productShowcaseSection" id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
              <ol>
                <li>
                  Download and install botorch from the Github repo.
                </li>
                <li>
                  Download Ax from the Github repo and install with pip:
                  <MarkdownBlock>
                    {bash`cd ax && pip3 install -e .`}
                  </MarkdownBlock>
                </li>
                <li>
                  Run an optimization:
                  <MarkdownBlock>{codeExample}</MarkdownBlock>
                </li>
              </ol>
          </Container>
        </div>
    );

    const Features = () => (
      <div
        className="productShowcaseSection"
        style={{textAlign: 'center'}}>
        <h2>Key Features</h2>
      <Block layout="fourColumn">
        {[
          {
            content: (
              'Easy to plug in new algorithms and use the library across ' +
              'different domains.'
            ),
            image: `${baseUrl}img/th-large-solid.svg`,
            imageAlign: 'top',
            title: 'Modular',
          },
          {
            content: (
              'Field experiments require a range of considerations ' +
              'beyond standard optimization problems.'
            ),
            image: `${baseUrl}img/dice-solid.svg`,
            imageAlign: 'top',
            title: 'Supports A/B Tests',
          },
          {
            content: (
              'Support for enterprise-level experimentation ' +
              'and optimization management, including MySQL storage.'
            ),
            image: `${baseUrl}img/database-solid.svg`,
            imageAlign: 'top',
            title: 'Production-Ready',
          },
        ]}
      </Block>
      </div>
    );

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
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
