/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const tutorials = () => {
  const allTutorialMetadata = require('./tutorials.json');
  const tutorialsSidebar = [{
    type: 'category',
    label: 'Tutorials',
    collapsed: false,
    items: [
      {
        type: 'doc',
        id: 'tutorials/index',
        label: 'Overview',
      },
    ],
  },];
  for (var category in allTutorialMetadata) {
    const categoryItems = allTutorialMetadata[category];
    const items = [];
    categoryItems.map(item => {
      items.push({
        type: 'doc',
        label: item.title,
        id: `tutorials/${item.id}/index`,
      });
    });

    tutorialsSidebar.push({
      type: 'category',
      label: category,
      items: items,
    });
  }
  return tutorialsSidebar;
};

export default {
  docs: {
    "Introduction": ["why-ax"],
    "Getting Started": ["installation", "api", "glossary"],
    "Algorithms": ["bayesopt", "banditopt"],
    "Components": ["core", "trial-evaluation", "data", "models", "storage"],
  },
  tutorials: tutorials(),
};
