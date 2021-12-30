/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

const baseUrl = '/';

// If true, include Algolia search bar when building site
// Note: this setting is toggled to false by publish_site.sh script, so
// it should not be renamed without modifying that script.
const includeAlgolia = true;

// List of projects/orgs using your project for the users page.
const users = [];

const siteConfig = {
  title: 'Ax',
  tagline: 'Adaptive Experimentation Platform',
  url: 'https://ax.dev/',
  baseUrl: baseUrl,

  // Used for publishing and more
  projectName: 'Ax',
  organizationName: 'facebook',

  // Google analytics
  gaTrackingId: 'UA-139570076-1',

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    {doc: 'why-ax', label: 'Docs'},
    {href: `${baseUrl}tutorials/`, label: 'Tutorials'},
    {href: `${baseUrl}api/`, label: 'API'},
    // Search can be enabled when site is online
    // {search: true},
    {href: 'https://github.com/facebook/Ax', label: 'GitHub'},
  ],

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */
  headerIcon: 'img/ax_lockup_white.svg',
  footerIcon: 'img/ax.svg',
  favicon: 'img/favicon.png',

  /* Colors for website */
  colors: {
    primaryColor: '#1F2833',
    secondaryColor: '#C5C6C7',
  },

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: 'default',
  },

  // Custom scripts that are placed in <head></head> of each page
  scripts: [
    'https://cdn.plot.ly/plotly-latest.min.js',
    `${baseUrl}js/plotUtils.js`,
    'https://buttons.github.io/buttons.js',
    `${baseUrl}js/mathjax.js`,
    'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML',
  ],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // Use .html extensions for paths.
  cleanUrl: false,

  // Open Graph and Twitter card images.
  ogImage: 'img/ax.svg',
  twitterImage: 'img/ax.svg',

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,

  // enable scroll to top button a the bottom of the site
  scrollToTop: true,

  wrapPagesHTML: true,
};

if (includeAlgolia == true) {
  siteConfig['algolia'] = {
    apiKey: '467d4f1f6cace3ecb36ab551cb44905b',
    indexName: 'ax',
    algoliaOptions: {}, // Optional, if provided by Algolia
  };
}

module.exports = siteConfig;
