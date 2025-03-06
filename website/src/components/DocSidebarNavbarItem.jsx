/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react";
import { useActiveDocContext } from '@docusaurus/plugin-content-docs/client';
import DocSidebarNavbarItem from '@theme-original/NavbarItem/DocSidebarNavbarItem';


/* Custom implementation of DocSidebarNavbarItem that supports conditionally
 * rendering based on the currently selected docs version.
 */
export default function ConditionalDocSidebarNavbarItem(props) {
  const { ignoreVersions, ...restOfProps } = props;

  const { activeVersion } = useActiveDocContext(props.docsPluginId);
  if (activeVersion === undefined || ignoreVersions.indexOf(activeVersion.name) !== -1) {
    return null;
  }

  return <DocSidebarNavbarItem {...restOfProps} />;
}
