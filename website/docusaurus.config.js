import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

module.exports={
  "title": "Ax",
  "tagline": "Adaptive Experimentation Platform",
  "url": "http://cristianlara.me",
  "baseUrl": "/Ax/",
  "organizationName": "cristianlara",
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
  "onBrokenLinks": "log",
  "onBrokenMarkdownLinks": "log",
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
    "navbar": {
      "title": "Ax",
      "logo": {
        "src": "img/ax.svg",
      },
      "items": [
        {
          "to": "docs/why-ax",
          "label": "Docs",
          "activeBaseRegex": "docs\/(?!tutorials\/)", // ignore tutorials subdirectory
          "position": "left"
        },
        {
          "to": "docs/tutorials/",
          "label": "Tutorials",
          "position": "left"
        },
        {
          "href": "/Ax/api/",
          "label": "API",
          "position": "left",
          "target": "_blank",
        },
        {
          "href": "https://github.com/cristianlara/Ax",
          "className": "header-github-link",
          "aria-label": "GitHub",
          "position": "right"
        },
        {
          to: 'blog',
          label: 'Blog',
          position: 'left'
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
              href: 'https://github.com/cristianlara/ax',
            },
            {
              label: 'Blog',
              href: 'https://github.com/cristianlara/ax/blog',
            },
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
      appId: '8X3RTPIIL1',
      apiKey: 'df847ece5bb2b557f7f1ce9f73f32387',
      indexName: 'cristianlara',
    },
  }
}
