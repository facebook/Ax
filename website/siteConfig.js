/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

// List of projects/orgs using your project for the users page.
const users = [];

const siteConfig = {
  title: 'Ax', // Title for your website.
  tagline: 'Automated optimization and experimentation',
  url: 'https://ax-platform.org', // Your website URL
  baseUrl: '/', // Base URL for your project */
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',

  // Used for publishing and more
  projectName: 'Ax',
  organizationName: 'facebook',
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    {doc: 'why-ax', label: 'Docs'},
    {href: '/tutorials/index', label: 'Tutorials'},
    {href: '/api/index', label: 'API'},
    // Search can be enabled when site is online
    // {search: true},
    {href: 'https://github.com/facebook/Ax', label: 'GitHub'},
  ],

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */
  headerIcon: null,
  footerIcon: null,
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
    'https://buttons.github.io/buttons.js',
    '/js/mathjax.js',
    'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML',
  ],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/docusaurus.png',
  twitterImage: 'img/docusaurus.png',

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,

  wrapPagesHTML: true,
};

module.exports = siteConfig;
