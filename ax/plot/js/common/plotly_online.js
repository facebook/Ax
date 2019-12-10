/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

requirejs.config({
  paths: {
    plotly: ['https://cdn.plot.ly/plotly-latest.min'],
  },
});
if (!window.Plotly) {
  require(['plotly'], function(plotly) {
    window.Plotly = plotly;
  });
}
