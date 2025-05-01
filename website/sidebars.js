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
  const tutorialsSidebar = [];
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
      collapsed: false,
      items: items,
    });
  }
  return tutorialsSidebar;
};

export default {
  docs: [
    {
      type: 'category',
      label: 'Introduction',
      collapsed: false,
      items: ['why-ax', 'installation', 'intro-to-ae', 'intro-to-bo'],
    },
    {
      type: 'category',
      label: 'Extending Ax',
      collapsed: false,
      items: ['analyses'],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: ['glossary'],
    },
  ],
  tutorials: [
    ...tutorials(),
    {
      type: 'category',
      label: 'Recipes',
      collapsed: false,
      items: [
        'custom-trials',
        'recipes/tracking-metrics',
        'recipes/experiment-to-json',
        'recipes/experiment-to-sqlite',
        'recipes/multi-objective-optimization',
        'recipes/scalarized-objective',
        'recipes/outcome-constraints',
      ],
    },
  ],
};
