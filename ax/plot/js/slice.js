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

// format data
const res = relativize_data(f, sd, rel, arm_data, metric);
const f_final = res[0];
const sd_final = res[1];

// get data for standard deviation fill plot
const sd_upper = [];
const sd_lower = [];
for (let i = 0; i < sd.length; i++) {
  sd_upper.push(f_final[i] + 2 * sd_final[i]);
  sd_lower.push(f_final[i] - 2 * sd_final[i]);
}
const grid_rev = copy_and_reverse(grid);
const sd_lower_rev = copy_and_reverse(sd_lower);
const sd_x = grid.concat(grid_rev);
const sd_y = sd_upper.concat(sd_lower_rev);

// get data for observed arms and error bars
const arm_x = [];
const arm_y = [];
const arm_sem = [];
fit_data.forEach(row => {
    parameters = arm_name_to_parameters[row["arm_name"]];
    plot = true;
    Object.keys(setx).forEach(p => {
      if (p !== param && parameters[p] !== setx[p]) {
        plot = false;
      }
    });
    if (plot === true) {
        arm_x.push(parameters[param]);
        arm_y.push(row["mean"]);
        arm_sem.push(row["sem"]);
    }
});

const arm_res = relativize_data(
  arm_y, arm_sem, rel, arm_data, metric
);
const arm_y_final = arm_res[0];
const arm_sem_final = arm_res[1].map(x => x * 2);

// create traces
const f_trace = {
  x: grid,
  y: f_final,
  showlegend: false,
  hoverinfo: 'x+y',
  line: {
    color: 'rgba(128, 177, 211, 1)',
  },
};

const arms_trace = {
  x: arm_x,
  y: arm_y_final,
  mode: 'markers',
  error_y: {
    type: 'data',
    array: arm_sem_final,
    visible: true,
    color: 'black',
  },
  line: {
    color: 'black',
  },
  showlegend: false,
  hoverinfo: 'x+y',
};

const sd_trace = {
    x: sd_x,
    y: sd_y,
    fill: 'toself',
    fillcolor: 'rgba(128, 177, 211, 0.2)',
    line: {
      color: 'transparent',
    },
    showlegend: false,
    hoverinfo: 'none',
};

traces = [
  sd_trace,
  f_trace,
  arms_trace,
];

// iterate over out-of-sample arms
let i = 1;
Object.keys(arm_data['out_of_sample']).forEach(generator_run_name => {
  const ax = [];
  const ay = [];
  const asem = [];
  const atext = [];

  Object.keys(arm_data['out_of_sample'][generator_run_name]).forEach(arm_name => {
    const parameters = arm_data[
      'out_of_sample'][generator_run_name][arm_name]['parameters'];
    plot = true;
    Object.keys(setx).forEach(p => {
      if (p !== param && parameters[p] !== setx[p]) {
        plot = false;
      }
    });
    if (plot === true) {
      ax.push(parameters[param]);
      ay.push(
        arm_data['out_of_sample'][generator_run_name][arm_name]['y_hat'][metric]
      );
      asem.push(
        arm_data['out_of_sample'][generator_run_name][arm_name]['se_hat'][metric]
      );
      atext.push('<em>Candidate ' + arm_name + '</em>');
    }
  });

  const out_of_sample_arm_res = relativize_data(
    ay, asem, rel, arm_data, metric
  );
  const ay_final = out_of_sample_arm_res[0];
  const asem_final = out_of_sample_arm_res[1].map(x => x * 2);

  traces.push({
    hoverinfo: 'text',
    legendgroup: generator_run_name,
    marker: {color: 'black', symbol: i, opacity: 0.5},
    mode: 'markers',
    error_y: {
      type: 'data',
      array: asem_final,
      visible: true,
      color: 'black',
    },
    name: generator_run_name,
    text: atext,
    type: 'scatter',
    xaxis: 'x',
    x: ax,
    yaxis: 'y',
    y: ay_final,
  });

  i += 1;
});

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
