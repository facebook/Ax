/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary');

const Container = CompLibrary.Container;

const CWD = process.cwd();

const versions = require(`${CWD}/_versions.json`);
// Sort the versions, handling the version numbers and extra characters
versions.sort(function(a, b) {
  a = a.replace("v", "");
  b = b.replace("v", "");
  var aArr = a.split(".");
  var bArr = b.split(".");
  if (aArr.len !== bArr.len) {
    throw 'Version formats do not match';
  }

  var aInt, bInt;
  for (var i = 0 ; i < aArr.length; i++) {
    aInt = parseInt(aArr[i]);
    bInt = parseInt(bArr[i]);
    if (aInt === bInt) {
      continue;
    }
    return aInt - bInt;
  }
  return 0;
}).reverse();


function Versions(props) {
  const {config: siteConfig} = props;
  const baseUrl = siteConfig.baseUrl;
  const latestVersion = versions[0];
  return (
    <div className="docMainWrapper wrapper">
      <Container className="mainContainer versionsContainer">
        <div className="post">
          <header className="postHeader">
            <h1>{siteConfig.title} Versions</h1>
          </header>

          <table className="versions">
            <tbody>
              <tr>
                <th>Version</th>
                <th>Install with</th>
                <th>Documentation</th>
              </tr>
              <tr>
                <td>{`stable (${latestVersion})`}</td>
                <td>
                  <code>pip3 install ax-platform</code>
                </td>
                <td>
                  <a href={`${baseUrl}index.html`}>stable</a>
                </td>
              </tr>
              <tr>
                <td>
                  {'latest'}
                  {' (main)'}
                </td>
                <td>
                  <code>
                    pip3 install git+ssh://git@github.com/facebook/Ax.git
                  </code>
                </td>
                <td>
                  <a href={`${baseUrl}versions/latest/index.html`}>latest</a>
                </td>
              </tr>
            </tbody>
          </table>

          <h3 id="archive">Past Versions</h3>
          <table className="versions">
            <tbody>
              {versions.map(
                version =>
                  version !== latestVersion && (
                    <tr key={version}>
                      <th>{version}</th>
                      <td>
                        <a href={`${baseUrl}versions/${version}/index.html`}>
                          Documentation
                        </a>
                      </td>
                    </tr>
                  ),
              )}
            </tbody>
          </table>
        </div>
      </Container>
    </div>
  );
}

module.exports = Versions;
