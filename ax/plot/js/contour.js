/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

const arm_data = {{arm_data}};
const density = {{density}};
const grid_x = {{grid_x}};
const grid_y = {{grid_y}};
const f = {{f}};
const lower_is_better = {{lower_is_better}};
const metric = {{metric}};
const rel = {{rel}};
const sd = {{sd}};
const xvar = {{xvar}};
const yvar = {{yvar}};
const x_is_log = {{x_is_log}};
const y_is_log = {{y_is_log}};

const GREEN_SCALE = {{green_scale}};
const GREEN_PINK_SCALE = {{green_pink_scale}};
const BLUE_SCALE = {{blue_scale}};

// format data
const res = relativize_data(f, sd, rel, arm_data, metric);
const f_final = res[0];
const sd_final = res[1];

// calculate max of abs(outcome), used for colorscale
const f_absmax = Math.max(Math.abs(Math.min(...f_final)), Math.max(...f_final));

// transform to nested array
var f_plt = [];
while(f_final.length) f_plt.push(f_final.splice(0, density));
var sd_plt = [];
while(sd_final.length) sd_plt.push(sd_final.splice(0, density));

// create traces
const CONTOUR_CONFIG = {
  autocolorscale: false,
  autocontour: true,
  contours: {
    coloring: 'heatmap',
  },
  hoverinfo: 'x+y+z',
  ncontours: density / 2,
  type: 'contour',
  x: grid_x,
  y: grid_y,
};

let f_scale;
if (rel === true) {
  f_scale = lower_is_better === true
    ? GREEN_PINK_SCALE.reverse()
    : GREEN_PINK_SCALE;
} else {
  f_scale = GREEN_SCALE;
}

const f_trace = {
  colorbar: {
    x: 0.45,
    y: 0.5,
    ticksuffix: rel === true ? '%' : '',
    tickfont: {
      size: 8,
    },
  },
  colorscale: f_scale.map(
    (v, i) => [i / (f_scale.length - 1), rgb(v)]
  ),
  xaxis: 'x',
  yaxis: 'y',
  z: f_plt,
  // zmax and zmin are ignored if zauto is true
  zauto: !rel,
  zmax: f_absmax,
  zmin: -f_absmax,
};

const sd_trace = {
  colorbar: {
      x: 1,
      y: 0.5,
      ticksuffix: rel === true ? '%' : '',
      tickfont: {
        size: 8,
      },
  },
  colorscale: BLUE_SCALE.map(
    (v, i) => [i / (BLUE_SCALE.length - 1), rgb(v)]
  ),
  xaxis: 'x2',
  yaxis: 'y2',
  z: sd_plt,
};

Object.keys(CONTOUR_CONFIG).forEach(key => {
  f_trace[key] = CONTOUR_CONFIG[key];
  sd_trace[key] = CONTOUR_CONFIG[key];
});

// get in-sample arms
const arm_x = [];
const arm_y = [];
const arm_text = [];

Object.keys(arm_data['in_sample']).forEach(arm_name => {
  arm_x.push(arm_data['in_sample'][arm_name]['parameters'][xvar]);
  arm_y.push(arm_data['in_sample'][arm_name]['parameters'][yvar]);
  arm_text.push(arm_name);
});

// configs for in-sample arms
const base_in_sample_arm_config = {
  hoverinfo: 'text',
  legendgroup: 'In-sample',
  marker: {color: 'black', symbol: 1, opacity: 0.5},
  mode: 'markers',
  name: 'In-sample',
  text: arm_text,
  type: 'scatter',
  x: arm_x,
  y: arm_y,
};

const f_in_sample_arm_trace = {
  xaxis: 'x',
  yaxis: 'y',
};

const sd_in_sample_arm_trace = {
  showlegend: false,
  xaxis: 'x2',
  yaxis: 'y2',
};

Object.keys(base_in_sample_arm_config).forEach(key => {
  f_in_sample_arm_trace[key] = base_in_sample_arm_config[key];
  sd_in_sample_arm_trace[key] = base_in_sample_arm_config[key];
});

const traces = [
  f_trace,
  sd_trace,
  f_in_sample_arm_trace,
  sd_in_sample_arm_trace,
];

// start symbol at 2 for candidate markers
let i = 2;

// iterate over out-of-sample arms
Object.keys(arm_data['out_of_sample']).forEach(generator_run_name => {
  const ax = [];
  const ay = [];
  const atext = [];

  Object.keys(arm_data['out_of_sample'][generator_run_name]).forEach(arm_name => {
    ax.push(
      arm_data['out_of_sample'][generator_run_name][arm_name]['parameters'][xvar]
    );
    ay.push(
      arm_data['out_of_sample'][generator_run_name][arm_name]['parameters'][yvar]
    );
    atext.push('<em>Candidate ' + arm_name + '</em>');
  });

  traces.push({
    hoverinfo: 'text',
    legendgroup: generator_run_name,
    marker: {color: 'black', symbol: i, opacity: 0.5},
    mode: 'markers',
    name: generator_run_name,
    text: atext,
    type: 'scatter',
    xaxis: 'x',
    x: ax,
    yaxis: 'y',
    y: ay,
  });
  traces.push({
    hoverinfo: 'text',
    legendgroup: generator_run_name,
    marker: {color: 'black', symbol: i, opacity: 0.5},
    mode: 'markers',
    name: 'In-sample',
    showlegend: false,
    text: atext,
    type: 'scatter',
    x: ax,
    xaxis: 'x2',
    y: ay,
    yaxis: 'y2',
  });
  i += 1;
});

// layout
const xrange = axis_range(grid_x, x_is_log);
const yrange = axis_range(grid_y, y_is_log);

const xtype = x_is_log ? 'log' : 'linear';
const ytype = y_is_log ? 'log' : 'linear';

const layout = {
  autosize: false,
    margin: {
      l: 35,
      r: 35,
      t: 35,
      b: 100,
      pad: 0,
  },
  annotations: [
    {
      font: {size: 14},
      showarrow: false,
      text: 'Mean',
      x: 0.25,
      xanchor: 'center',
      xref: 'paper',
      y: 1,
      yanchor: 'bottom',
      yref: 'paper',
    },
    {
      font: {size: 14},
      showarrow: false,
      text: 'Standard Error',
      x: 0.8,
      xanchor: 'center',
      xref: 'paper',
      y: 1,
      yanchor: 'bottom',
      yref: 'paper',
    },
  ],
  hovermode: 'closest',
  legend: {orientation: 'h', x: 0, y: -0.25},
  height: 450,
  width: 950,
  xaxis: {
    anchor: 'y',
    autorange: false,
    domain: [0.05, 0.45],
    exponentformat: 'e',
    range: xrange,
    tickfont: {size: 11},
    tickmode: 'auto',
    title: xvar,
    type: xtype,
  },
  xaxis2: {
    anchor: 'y2',
    autorange: false,
    domain: [0.60, 1],
    exponentformat: 'e',
    range: xrange,
    tickfont: {size: 11},
    tickmode: 'auto',
    title: xvar,
    type: xtype,
  },
  yaxis: {
    anchor: 'x',
    autorange: false,
    domain: [0, 1],
    exponentformat: 'e',
    range: yrange,
    tickfont: {size: 11},
    tickmode: 'auto',
    title: yvar,
    type: ytype,
  },
  yaxis2: {
    anchor: 'x2',
    autorange: false,
    domain: [0, 1],
    exponentformat: 'e',
    range: yrange,
    tickfont: {size: 11},
    tickmode: 'auto',
    type: ytype,
  },
};

Plotly.newPlot({{id}}, traces, layout, {showLink: false});
