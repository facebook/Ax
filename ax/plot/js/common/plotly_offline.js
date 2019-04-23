/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

if (!window.Plotly) {
  define('plotly', function(require, exports, module) {
    {{library}}
  });
  require(['plotly'], function(Plotly) {
    window.Plotly = Plotly;
  });
}
