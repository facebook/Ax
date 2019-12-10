/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

if (!window.Plotly) {
  define('plotly', function(require, exports, module) {
    {{library}}
  });
  require(['plotly'], function(Plotly) {
    window.Plotly = Plotly;
  });
}
