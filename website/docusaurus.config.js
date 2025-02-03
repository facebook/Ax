/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

module.exports={
  "title": "Ax",
  "tagline": "Adaptive Experimentation Platform",
  "url": "https://ax.dev",
  "baseUrl": "/",
  "organizationName": "facebook",
  "projectName": "Ax",
  "scripts": [
    "https://cdn.plot.ly/plotly-latest.min.js",
    "/Ax/js/plotUtils.js",
    "https://buttons.github.io/buttons.js",
    'https://cdn.bokeh.org/bokeh/release/bokeh-2.4.2.min.js',
    'https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.4.2.min.js',
  ],
  "favicon": "img/favicon.png",
  "customFields": {
    "users": [],
    "wrapPagesHTML": true
  },
  "onBrokenLinks": "throw",
  "onBrokenMarkdownLinks": "warn",
  "future": {
    "experimental_faster": true,
  },
  "presets": [
    [
      "@docusaurus/preset-classic",
      {
        "docs": {
          "showLastUpdateAuthor": true,
          "showLastUpdateTime": true,
          "path": "../docs",
          "sidebarPath": "../website/sidebars.js",
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        "blog": {},
        "theme": {
          "customCss": "src/css/customTheme.css"
        },
        "gtag": {
          "trackingID": "UA-139570076-1"
        }
      }
    ]
  ],
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
  "plugins": [
    [
      "@docusaurus/plugin-client-redirects",
      {
        "fromExtensions": [
          "html"
        ]
      }
    ],
  ],
  "themeConfig": {
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.oneDark,
    },
    "navbar": {
      "title": "Ax",
      "hideOnScroll": true,
      "logo": {
        "src": "img/ax.svg",
      },
      "items": [
        {
          "type": "docSidebar",
          "sidebarId": "docs",
          "label": "Docs",
          "position": "left"
        },
        {
          "type": "docSidebar",
          "sidebarId": "tutorials",
          "label": "Tutorials",
          "position": "left"
        },
        {
          "href": "https://ax.readthedocs.io/",
          "label": "API",
          "position": "left",
          "target": "_blank",
        },
        {
          type: 'docsVersionDropdown',
          position: 'right',
          dropdownItemsAfter: [
              {
                type: 'html',
                value: '<hr class="margin-vert--sm">',
              },
              {
                type: 'html',
                className: 'margin-horiz--sm text--bold',
                value: '<small>Archived versions<small>',
              },
              {
                href: 'https://archive.ax.dev/versions.html',
                label: '0.x.x',
              },
            ],
        },
        {
          "href": "https://github.com/facebook/Ax",
          "className": "header-github-link",
          "aria-label": "GitHub",
          "position": "right"
        }
      ]
    },
    "docs": {
        "sidebar": {
          autoCollapseCategories: true,
          hideable: true,
        },
    },
    "image": "img/ax.svg",
    "footer": {
      style: 'dark',
      "logo": {
        alt: "Ax",
        "src": "img/meta_opensource_logo_negative.svg",
      },
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: 'docs/why-ax',
            },
          ],
        },
        {
          title: 'Social',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/facebook/ax',
            }
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc.`,
    },
    algolia: {
      appId: 'O2Q3QH4SYH',
      apiKey: '330b76ae9b20640dacf7ef3e1256f584',
      indexName: 'ax',
    },
  }
}
