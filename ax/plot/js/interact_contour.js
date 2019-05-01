/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

const arm_data = {{arm_data}};
const density = {{density}};
const grid_dict = {{grid_dict}};
const f_dict = {{f_dict}};
const lower_is_better = {{lower_is_better}};
const metric = {{metric}};
const rel = {{rel}};
const sd_dict = {{sd_dict}};
const is_log_dict = {{is_log_dict}};
const param_names = {{param_names}};

const GREEN_SCALE = {{green_scale}};
const GREEN_PINK_SCALE = {{green_pink_scale}};
const BLUE_SCALE = {{blue_scale}};

const MAX_PARAM_LENGTH = 40;
function short_name(param_name) {
  if (param_name.length < MAX_PARAM_LENGTH) {
    return param_name;
  }

  // Try to find a canonical prefix
  prefix = param_name.split(/ |_|:/)[0];
  if (prefix.length > 10) {
    prefix = param_name.substring(0, 10);
  }
  suffix = param_name.substring(
    param_name.length - (MAX_PARAM_LENGTH - prefix.length - 3)
  );
  return prefix.concat("...").concat(suffix);
}

let f_scale;
if (rel === true) {
  f_scale = lower_is_better === true
    ? GREEN_PINK_SCALE.reverse()
    : GREEN_PINK_SCALE;
} else {
  f_scale = GREEN_SCALE;
}
let insample_arm_text = Object.keys(arm_data['in_sample']);

let f_contour_trace_base = {
  colorbar: {
    x: 0.45,
    y: 0.5,
    ticksuffix: rel === true ? '%' : '',
    tickfont: {
      size: 8,
    },
    len: 0.875,
  },
  colorscale: f_scale.map(
    (v, i) => [i / (f_scale.length - 1), rgb(v)]
  ),
  xaxis: 'x',
  yaxis: 'y',
  zauto: !rel,

  // shared with sd_contour
  autocolorscale: false,
  autocontour: true,
  contours: {
    coloring: 'heatmap',
  },
  hoverinfo: 'x+y+z',
  ncontours: density / 2,
  type: 'contour',
};

let sd_contour_trace_base = {
  colorbar: {
      x: 1,
      y: 0.5,
      ticksuffix: rel === true ? '%' : '',
      tickfont: {
        size: 8,
      },
      len: 0.875,
  },
  colorscale: BLUE_SCALE.map(
    (v, i) => [i / (BLUE_SCALE.length - 1), rgb(v)]
  ),
  xaxis: 'x2',
  yaxis: 'y2',

  // shared with f_contour
  autocolorscale: false,
  autocontour: true,
  contours: {
    coloring: 'heatmap',
  },
  hoverinfo: 'x+y+z',
  ncontours: density / 2,
  type: 'contour',
};

let insample_param_values = {};
param_names.forEach(param_name => {
    insample_param_values[param_name] = [];
    Object.keys(arm_data['in_sample']).forEach(arm_name => {
        insample_param_values[param_name].push(
            arm_data['in_sample'][arm_name]['parameters'][param_name]
        );
    });
});

let out_of_sample_param_values = {};
param_names.forEach(param_name => {
  out_of_sample_param_values[param_name] = {};
  Object.keys(arm_data['out_of_sample']).forEach(generator_run_name => {
    out_of_sample_param_values[param_name][generator_run_name] = [];
    Object.keys(arm_data['out_of_sample'][generator_run_name]).forEach(arm_name => {
      out_of_sample_param_values[param_name][generator_run_name].push(
        arm_data['out_of_sample'][generator_run_name][arm_name]['parameters'][param_name]
      );
    });
  });
});

let out_of_sample_arm_text = {};
Object.keys(arm_data['out_of_sample']).forEach(generator_run_name => {
  out_of_sample_arm_text[generator_run_name] =
    Object.keys(arm_data['out_of_sample'][generator_run_name]).map(
      arm_name => '<em>Candidate ' + arm_name + '</em>'
    );
});

// Number of traces for each pair of parameters
let trace_cnt = 4 + (Object.keys(arm_data['out_of_sample']).length * 2);

let xbuttons = [];
let ybuttons = [];

for (var xvar_idx in param_names) {
  let xvar = param_names[xvar_idx];
  let xbutton_data_args = {
    "x": [],
    "y": [],
    "z": [],
  };
  for (var yvar_idx in param_names) {
    let yvar = param_names[yvar_idx];
    const res = relativize_data(
      f_dict[xvar][yvar], sd_dict[xvar][yvar], rel, arm_data, metric
    );
    const f_final = res[0];
    const sd_final = res[1];
    // transform to nested array
    var f_plt = [];
    while(f_final.length) f_plt.push(f_final.splice(0, density));
    var sd_plt = [];
    while(sd_final.length) sd_plt.push(sd_final.splice(0, density));

    // grid + in-sample
    xbutton_data_args["x"] = xbutton_data_args["x"].concat(
      [grid_dict[xvar], grid_dict[xvar],
      insample_param_values[xvar], insample_param_values[xvar]]
    );
    xbutton_data_args["y"] = xbutton_data_args["y"].concat(
      [grid_dict[yvar], grid_dict[yvar],
      insample_param_values[yvar], insample_param_values[yvar]]
    );
    xbutton_data_args["z"] = xbutton_data_args["z"].concat(
      [f_plt, sd_plt, [], []]
    );

    Object.keys(out_of_sample_param_values[xvar]).forEach(
      generator_run_name => {
        let generator_run_x_vals = out_of_sample_param_values[xvar][
          generator_run_name
        ];
        xbutton_data_args["x"] = xbutton_data_args["x"].concat(
          [generator_run_x_vals, generator_run_x_vals]
        );
    });
    Object.keys(out_of_sample_param_values[yvar]).forEach(
      generator_run_name => {
        let generator_run_y_vals = out_of_sample_param_values[yvar][
          generator_run_name
        ];
        xbutton_data_args["y"] = xbutton_data_args["y"].concat(
          [generator_run_y_vals, generator_run_y_vals]
        );
    });
    Object.keys(out_of_sample_param_values[yvar]).forEach(
      generator_run_name => {
        xbutton_data_args["z"] = xbutton_data_args["z"].concat([[], []]);
    });
  }
  xbutton_args = [
    xbutton_data_args,
    {
      "xaxis.title": short_name(xvar),
      "xaxis2.title": short_name(xvar),
      "xaxis.range": axis_range(grid_dict[xvar], is_log_dict[xvar]),
      "xaxis2.range": axis_range(grid_dict[xvar], is_log_dict[xvar]),
    }
  ];
  xbuttons.push({
    "args": xbutton_args,
    "label": xvar,
    "method": "update",
  });
}

// No y button for first param so initial value is sane
for (var y_idx = 1; y_idx < param_names.length; y_idx++) {
  visible = new Array(param_names.length * trace_cnt);
  visible.fill(false).fill(true, y_idx * trace_cnt, (y_idx + 1) * trace_cnt);
  let y_param = param_names[y_idx];
  ybuttons.push({
    "args": [
      {"visible": visible},
      {
        "yaxis.title": short_name(y_param),
        "yaxis.range": axis_range(grid_dict[y_param], is_log_dict[y_param]),
        "yaxis2.range": axis_range(grid_dict[y_param], is_log_dict[y_param]),
      }
    ],
    "label": param_names[y_idx],
    "method": "update",
  });
}

// calculate max of abs(outcome), used for colorscale
// TODO(T37079623) Make this work for relative outcomes
// let f_absmax = Math.max(Math.abs(Math.min(...f_final)), Math.max(...f_final))

traces = [];
let xvar = param_names[0];
let base_in_sample_arm_config;

// start symbol at 2 for out-of-sample candidate markers
let i = 2;

for (var yvar_idx in param_names) {
    let cur_visible = yvar_idx == 1;
    let yvar = param_names[yvar_idx];
    let f_start = xbuttons[0]["args"][0]["z"][trace_cnt * yvar_idx];
    let sd_start = xbuttons[0]["args"][0]["z"][trace_cnt * yvar_idx + 1];

    // create traces
    const f_trace = {
      x: grid_dict[xvar],
      y: grid_dict[yvar],
      z: f_start,
      visible: cur_visible,
      // zmax and zmin are ignored if zauto is true
      // zmax: f_absmax,
      // zmin: -f_absmax,
    };

    Object.keys(f_contour_trace_base).forEach(key => {
      f_trace[key] = f_contour_trace_base[key];
    });

    const sd_trace = {
      x: grid_dict[xvar],
      y: grid_dict[yvar],
      z: sd_start,
      visible: cur_visible,
    };

    Object.keys(sd_contour_trace_base).forEach(key => {
      sd_trace[key] = sd_contour_trace_base[key];
    });

    const f_in_sample_arm_trace = {
      xaxis: 'x',
      yaxis: 'y',
    };

    const sd_in_sample_arm_trace = {
      showlegend: false,
      xaxis: 'x2',
      yaxis: 'y2',
    };
    base_in_sample_arm_config = {
      hoverinfo: 'text',
      legendgroup: 'In-sample',
      marker: {color: 'black', symbol: 1, opacity: 0.5},
      mode: 'markers',
      name: 'In-sample',
      text: insample_arm_text,
      type: 'scatter',
      visible: cur_visible,
      x: insample_param_values[xvar],
      y: insample_param_values[yvar],
    };

    Object.keys(base_in_sample_arm_config).forEach(key => {
      f_in_sample_arm_trace[key] = base_in_sample_arm_config[key];
      sd_in_sample_arm_trace[key] = base_in_sample_arm_config[key];
    });
    traces = traces.concat([
      f_trace,
      sd_trace,
      f_in_sample_arm_trace,
      sd_in_sample_arm_trace,
    ]);
    // iterate over out-of-sample arms

    Object.keys(arm_data['out_of_sample']).forEach(generator_run_name => {
      traces.push({
        hoverinfo: 'text',
        legendgroup: generator_run_name,
        marker: {color: 'black', symbol: i, opacity: 0.5},
        mode: 'markers',
        name: generator_run_name,
        text: out_of_sample_arm_text[generator_run_name],
        type: 'scatter',
        xaxis: 'x',
        x: out_of_sample_param_values[xvar][generator_run_name],
        yaxis: 'y',
        y: out_of_sample_param_values[yvar][generator_run_name],
        visible: cur_visible,
      });
      traces.push({
        hoverinfo: 'text',
        legendgroup: generator_run_name,
        marker: {color: 'black', symbol: i, opacity: 0.5},
        mode: 'markers',
        name: 'In-sample',
        showlegend: false,
        text: out_of_sample_arm_text[generator_run_name],
        type: 'scatter',
        x: out_of_sample_param_values[xvar][generator_run_name],
        xaxis: 'x2',
        y: out_of_sample_param_values[yvar][generator_run_name],
        yaxis: 'y2',
        visible: cur_visible,
      });
      i += 1;
    });

}
// Initially visible yvar
let yvar = param_names[1];

// layout
const xrange = axis_range(grid_dict[xvar], is_log_dict[xvar]);
const yrange = axis_range(grid_dict[yvar], is_log_dict[yvar]);

const xtype = is_log_dict[xvar] ? 'log' : 'linear';
const ytype = is_log_dict[yvar] ? 'log' : 'linear';

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
    {
        "x": 0.26,
        "y": -0.26,
        "xref": "paper",
        "yref": "paper",
        "text": "x-param:",
        "showarrow": false,
        "yanchor": "top",
        "xanchor": "left",
    },
    {
        "x": 0.26,
        "y": -0.4,
        "xref": "paper",
        "yref": "paper",
        "text": "y-param:",
        "showarrow": false,
        "yanchor": "top",
        "xanchor": "left",
    },
  ],
  updatemenus: [
    {
        "x": 0.35,
        "y": -0.29,
        "buttons": xbuttons,
        "xanchor": "left",
        "yanchor": "middle",
        "direction": "up",
    },
    {
        "x": 0.35,
        "y": -0.43,
        "buttons": ybuttons,
        "xanchor": "left",
        "yanchor": "middle",
        "direction": "up",
    }
  ],
  hovermode: 'closest',
  legend: {orientation: 'v', x: 0, y: -0.2, yanchor: "top"},
  height: 525,
  width: 950,
  title: "Contours of metric " + metric,
  xaxis: {
    anchor: 'y',
    autorange: false,
    domain: [0.05, 0.45],
    exponentformat: 'e',
    range: xrange,
    tickfont: {size: 11},
    tickmode: 'auto',
    title: short_name(xvar),
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
    title: short_name(xvar),
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
    title: short_name(yvar),
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
