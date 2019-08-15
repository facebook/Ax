/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

const arm_data = {{arm_data}};
const arm_name_to_parameters = {{arm_name_to_parameters}};
const f = {{f}};
const fit_data = {{fit_data}};
const grid = {{grid}};
const metric = {{metric}};
const param = {{param}};
const rel = {{rel}};
const setx = {{setx}};
const sd = {{sd}};
const is_log = {{is_log}};

traces = slice_config_to_trace(
  arm_data,
  arm_name_to_parameters,
  f,
  fit_data,
  grid,
  metric,
  param,
  rel,
  setx,
  sd,
  is_log,
  true,
);

// layout
const xrange = axis_range(grid, is_log);
const xtype = is_log ? 'log' : 'linear';

layout = {
  hovermode: 'closest',
  xaxis: {
    anchor: 'y',
    autorange: false,
    exponentformat: 'e',
    range: xrange,
    tickfont: {size: 11},
    tickmode: 'auto',
    title: param,
    type: xtype,
  },
  yaxis: {
    anchor: 'x',
    tickfont: {size: 11},
    tickmode: 'auto',
    title: metric,
  },
};

Plotly.newPlot(
    {{id}},
    traces,
    layout,
    {showLink: false},
);
