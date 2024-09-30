/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import Link from '@docusaurus/Link';

const LinkButtons = ({githubUrl, colabUrl}) => {
  return (
    <div className="link-buttons">
      <Link to={githubUrl}>Open in GitHub</Link>
      <div></div>
      <Link to={colabUrl}>Run in Google Colab</Link>
    </div>
  );
};

export default LinkButtons;
