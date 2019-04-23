/*
 * Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
 */

const TOP_MARGIN = 40;
const BOTTOM_MARGIN = 50;
const RIGHT_MARGIN = 80;
const MIN_WIDTH = 750;

const ds = {{ds}};
const metrics = {{metrics}};
const available = {{available}};
const experiment_name = {{experiment_name}};
const is_range = {{is_range}};

const trace = {
    x: ds,
    y: metrics,
    z: available,
    type: 'heatmap',
    autocolorscale: false,
    hoverinfo: 'x+y',
    zmin: 0,
    zmax: 1,
    xgap: is_range === true ? 0 : 2,
    ygap: 2,
    showscale: false,
    colorscale: [
        [0, "#ffffff"],
        [0.5, "#cccccc"],
        [1, "#3b5998"],
    ],
};

// add some invisible traces for legend (since legend above is a continuous
// colorscale, not discrete)
const hidden_trace1 = {
  x: ['yes'],
  y: [0],
  marker: {color: '#3b5998'},
  name: is_range === true ? 'All metrics available' : 'Available',
  type: 'bar',
  visible: 'legendonly',
};

const data = [trace, hidden_trace1];

// only include metric available if using a plot that highlights the common
// range
if (is_range === true) {
  data.push({
    x: ['yes'],
    y: [0],
    marker: {color: '#cccccc'},
    name: 'Metric available',
    type: 'bar',
    visible: 'legendonly',
  });
}

data.push({
  x: ['no'],
  y: [0],
  marker: {color: '#ffffff', 'line': {'color': '#cccccc', 'width': 1}},
  name: 'Unavailable',
  type: 'bar',
  visible: 'legendonly',
});

const leftMargin = Math.max.apply(null, metrics.map(m => m.length)) * 7.75;

// if fewer than 4 metrics, make a bit taller so that the overall plot height
// allows for rendering of legend
const perMetricHeight = metrics.length > 4 ? 25 : 35;

const layout = {
  height:  perMetricHeight * metrics.length + TOP_MARGIN + BOTTOM_MARGIN,
  margin: {
    b: BOTTOM_MARGIN,
    l: leftMargin,
    t: TOP_MARGIN,
  },
  title: "Data availability by metric and ds for " + experiment_name,
  titlefont: {size: 12},
};

// calculate when we should force a minimum width
const width = 20 * ds.length + leftMargin + RIGHT_MARGIN;
if (width < MIN_WIDTH) {
  layout['width'] = MIN_WIDTH;
}

Plotly.newPlot({{id}}, data, layout, {showLink: false});

// this is a hacky way to disable legend toggling by hiding toggle layer
// and to remove transparency
$('#' + {{id}} + ' .legend .legendtoggle').hide();
$('#' + {{id}} + ' .legend .traces').css('opacity', 1);
