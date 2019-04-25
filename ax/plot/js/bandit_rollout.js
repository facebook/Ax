/*
 * Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
 */

const data = {{data}};
const categories = {{categories}};
const colors = {{colors}};

layout = {
    title: "Rollout Process<br>Bandit Weight Graph",
    xaxis: {
      title: "Rounds",
      zeroline: false,
      categoryorder: "array",
      categoryarray: categories
    },
    yaxis: {
      title: "Percent",
      showline: false
    },
    barmode: "stack",
    showlegend: false,
    margin: {
      r: 40
    }
}

bandit_config = {
    type: "bar",
    hoverinfo: "name+text",
    width: 0.5
}

bandits = [];
data.map(d => {
  bandits.push({
    ...bandit_config,
    ...d,
    marker: {
      color: colors[d.index % colors.length]
    }
  })
});

Plotly.newPlot({{id}}, bandits, layout, {showLink: false});
