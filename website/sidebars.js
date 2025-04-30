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
  const tutorialsSidebar = [
    {
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
    },
  ];
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
    Introduction: ['why-ax', 'installation', 'intro-to-ae', 'intro-to-bo'],
    'Extending Ax': ['analyses'],
    Reference: ['glossary'],
  },
  tutorials: tutorials(),
  recipes: {
    Recipes: [
      'recipes/index',
      'custom-trials',
      'recipes/tracking-metrics',
      'recipes/experiment-to-json',
      'recipes/experiment-to-sqlite',
      'recipes/multi-objective-optimization',
      'recipes/scalarized-objective',
      'recipes/outcome-constraints',
    ],
  },
};
