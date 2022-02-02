/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

window.MathJax = {
  tex2jax: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
  },
  // Center justify equations in code and markdown cells. Note that this
  // doesn't work with Plotly though, hence the !important declaratio
  // below.
  displayAlign: 'center',
  'HTML-CSS': {
    styles: {
      '.MathJax_Display': {margin: 0, 'text-align': 'center !important'},
    },
    linebreaks: {automatic: true},
  },
};
