/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

class Footer extends React.Component {

  docUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    const docsUrl = this.props.config.docsUrl;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    return `${baseUrl}${docsPart}${langPart}${doc}`;
  }

  pageUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    return baseUrl + (language ? `${language}/` : '') + doc;
  }

  render() {
    const currentYear = new Date().getFullYear();

    return (
      <footer className="nav-footer" id="footer">
        <section className="sitemap">
          <div className="footerSection">
            <h5>Legal</h5>
            <a
              href="https://opensource.facebook.com/legal/privacy/"
              target="_blank"
              rel="noreferrer noopener">
              Privacy
            </a>
            <a
              href="https://opensource.facebook.com/legal/terms/"
              target="_blank"
              rel="noreferrer noopener">
              Terms
            </a>
          </div>
        </section>
        <a
          href="https://code.facebook.com/projects/"
          target="_blank"
          rel="noreferrer noopener"
          className="fbOpenSource">
          <img
            src={`${this.props.config.baseUrl}img/oss_logo.png`}
            alt="Facebook Open Source"
            width="170"
            height="45"
          />
        </a>
        <section className="copyright">
          Copyright &copy; {currentYear} Meta Platforms, Inc.
        </section>
      </footer>
    );
  }
}

module.exports = Footer;
