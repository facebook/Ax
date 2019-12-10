/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

var css = document.createElement('style');
css.type = 'text/css';
css.innerHTML = "{{css}}";
document.getElementsByTagName("head")[0].appendChild(css);
