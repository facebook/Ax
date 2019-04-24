/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

const allocations = {{qe_allocations}};
const experiment_name = {{experiment_name}};
const exposures = {{cumulative_exposures}};
const gk_change_ds = {{gk_change_ds}};
const min_ds = {{min_ds}};
const max_ds = {{max_ds}};
const targeting_gk = {{targeting_gk}};

const aggregated = !Object.keys(exposures).includes('columns');
const exposure_traces = []

if (!aggregated) {
  // need to transpose arrays so that each nested array corresponds to
  // cumulative exposures per arm over ds
  const exposure_data = exposures['data'][0].map(function(col, i) {
    return exposures['data'].map(function(row) {
      return row[i];
    });
  });

  exposures['columns'].forEach((arm_name, i) => {
    exposure_traces.push({
      x: exposures['index'].map(ds => new Date(ds)),
      y: exposure_data[i],
      hoverinfo: 'x+y+text+name',
      name: arm_name,
      xaxis: 'x',
      yaxis: 'y',
      mode: 'lines+markers',
      type: 'scatter',
    });
  });
} else {
  exposure_traces.push({
    x: exposures['index'].map(ds => new Date(ds)),
    y: exposures['data'],
    hoverinfo: 'x+y+text',
    xaxis: 'x',
    yaxis: 'y',
    mode: 'lines+markers',
    type: 'scatter',
  });
}

const time_options = {
  weekday: 'long',
  year: 'numeric',
  month: 'long',
  day: 'numeric',
  hour: 'numeric',
  minute: 'numeric',
};

const shapes = [];
const annotations = [];

// if QE allocation data exists, plot segment + GK changes
if (allocations !== null) {
  const alloc_labels = allocations.map(el => {
    return 'Segment allocation changed from ' +
      el.prev_size + ' to ' +
      el.new_size + '<br />on ' +
      new Date(el.time).toLocaleString('en-US', time_options) +
      ' by ' + el.unixname;
  });

  exposure_traces.push({
    x: allocations.map(el => new Date(el.time)),
    y: allocations.map(el => el.new_size),
    line: {color: '#C0C0C0', shape: 'hv'},
    text: alloc_labels,
    hoverinfo: 'text',
    xaxis: 'x2',
    yaxis: 'y2',
    fill: 'tozeroy',
    marker: {size: Array(allocations.length).fill(10)},
    mode: 'lines+markers',
    type: 'scatter',
    showlegend: false,
  });

  // if experiment is still ongoing (not deallocated), continue trace to max_ds
  const last_new_size = allocations[allocations.length - 1]['new_size'];
  if (last_new_size != 0) {
    exposure_traces[exposure_traces.length - 1]['x'].push(new Date(max_ds));
    exposure_traces[exposure_traces.length - 1]['y'].push(last_new_size);
    exposure_traces[exposure_traces.length - 1]['marker']['size'].push(0);
  }

  const gk_uri = (
    "https://our.intern.facebook.com/intern/gatekeeper/projects/" +
    targeting_gk
  );

  gk_change_ds.forEach(ds => {
    shapes.push({
      line: {dash: 'dash'},
      type: 'line',
      x0: ds,
      x1: ds,
      xref: 'x',
      y0: 0,
      y1: 0.20,
      yref: 'paper',
    });
    annotations.push({
      showarrow: false,
      text: '<a href="' + gk_uri + '" target="_blank">GK</a>',
      x: ds,
      xref: 'x',
      y: 0.20,
      yref: 'paper',
    });
  });
}

// placement of first tick on x-axis
const tick0 = Math.min(...exposures['index'].map(d => new Date(d)));

// place ticks every 2 days (if < 1 month); else every 7 days
let dtick = 86400000.0 * 2;
if ((new Date(max_ds) - new Date(min_ds)) / 86400000.0 > 30) {
  dtick = 86400000.0 * 7;
}

const xaxis_config = {
  autorange: false,
  range:[min_ds, max_ds],
  autotick: false,
  tick0: tick0,
  tickformat:'%b %d',
  dtick: dtick,
};

// layout if allocation data exists
const layoutSegments = {
  hovermode: 'closest',
  yaxis: {
    domain: [0.35, 1],
    hoverformat: '.3s',
    rangemode: 'tozero',
    title: 'Cumulative Exposures',
  },
  xaxis: xaxis_config,
  xaxis2: {
    autorange: false,
    range: [min_ds, max_ds],
    anchor: 'y2',
    autotick: false,
    tick0: tick0,
    tickformat:'%b %d',
    dtick: dtick,
  },
  yaxis2: {domain: [0, 0.20], title: 'Segments'},
  showlegend: !aggregated,
  shapes: shapes,
  annotations: annotations,
  title: 'Cumulative exposures for ' + experiment_name,
};

// if just a single allocation, manually set ticks at 0 and new_size
if (allocations !== null && allocations.length === 1) {
  layoutSegments['yaxis2']['array'] = 'array';
  layoutSegments['yaxis2']['tickvals'] = [0, allocations[0]['new_size']];
}

// layout if no allocation data
const layoutNoSegments = {
  hovermode: 'closest',
  yaxis: {
    hoverformat: '.3s',
    title: 'Cumulative Exposures',
  },
  xaxis: xaxis_config,
  showlegend: !aggregated,
  title: 'Cumulative exposures for ' + experiment_name,
};

Plotly.newPlot(
  {{id}},
  exposure_traces,
  allocations !== null ? layoutSegments : layoutNoSegments,
  {showLink: false},
);
