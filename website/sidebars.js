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
      items: [
        'why-ax',
        'installation',
        'intro-to-ae',
        'intro-to-bo',
        'ae-vs-traditional',
      ],
    },
    {
      type: 'category',
      label: 'Internal Organization of Ax',
      collapsed: false,
      items: ['experiment', 'orchestration', 'generation_strategy'],
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
        'recipes/attach-trial',
        'recipes/existing-data',
        'recipes/tracking-metrics',
        'recipes/experiment-to-json',
        'recipes/experiment-to-sqlite',
        'recipes/multi-objective-optimization',
        'recipes/scalarized-objective',
        'recipes/outcome-constraints',
        'recipes/influence-gs-choice',
      ],
    },
  ],
};
