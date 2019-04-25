/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
