/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

const arm_data = {{arm_data}};
const arm_name_to_parameters = {{arm_name_to_parameters}};
const f = {{f}};
const fit_data = {{fit_data}};
const grid = {{grid}};
const metrics = {{metrics}}
const param = {{param}};
const rel = {{rel}};
const setx = {{setx}};
const sd = {{sd}};
const is_log = {{is_log}};

traces = [];
metric_cnt = metrics.length;

for (let i = 0; i < metric_cnt; i++) {
  cur_visible = i == 0;
  metric = metrics[i];
  traces = traces.concat(
    slice_config_to_trace(
      arm_data[metric],
      arm_name_to_parameters[metric],
      f[metric],
      fit_data[metric],
      grid,
      metric,
      param,
      rel,
      setx,
      sd[metric],
      is_log[metric],
      cur_visible,
    ),
  );
}

// layout
const xrange = axis_range(grid, is_log[metrics[0]]);
const xtype = is_log[metrics[0]] ? 'log' : 'linear';

let buttons = [];
for (let i = 0; i < metric_cnt; i++) {
  metric = metrics[i];
  let trace_cnt = 3 + Object.keys(arm_data[metric]['out_of_sample']).length * 2;
  visible = new Array(metric_cnt * trace_cnt);
  visible.fill(false).fill(true, i * trace_cnt, (i + 1) * trace_cnt);
  buttons.push({
    method: 'update',
    args: [{visible: visible}, {'yaxis.title': metric}],
    label: metric,
  });
}

const layout = {
  title: 'Predictions for a 1-d slice of the parameter space',
  annotations: [
    {
      showarrow: false,
      text: 'Choose metric:',
      x: 0.225,
      xanchor: 'center',
      xref: 'paper',
      y: 1.005,
      yanchor: 'bottom',
      yref: 'paper',
    },
  ],
  updatemenus: [
    {
      y: 1.1,
      x: 0.5,
      yanchor: 'top',
      buttons: buttons,
    },
  ],
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
    autorange: true,
    anchor: 'x',
    tickfont: {size: 11},
    tickmode: 'auto',
    title: metrics[0],
  },
};

Plotly.newPlot(
  {{id}},
  traces,
  layout,
  {showLink: false},
);
