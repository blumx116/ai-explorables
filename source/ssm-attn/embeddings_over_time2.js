const colors = ["#FAEDCB", "#C9E4DE", "#DBCDF0", "#FD8A8A"];
const n_axes = 3;
const n_tokens = 4;
const tokens_list = ["A", "B", "C", "✻"];

function axis_minmax(embeddings, axis) {
  var values = [];
  for (var i = 0; i < embeddings.length; i++) {
    values = [...values, ...embeddings[i][axis]];
  }
  return [Math.min(...values), Math.max(...values)];
}

function axis_config(embeddings, axis) {
  const [min, max] = axis_minmax(embeddings, axis);
  return {
    range: [min - 0.05, max + 0.05],
    tickmode: "linear",
    dtick: 1,
    tick0: Math.ceil(min - 0.05),
    autorange: false,
  };
}

function make_layout(embeddings) {
  return {
    title: {
      text: "Evolution of Token Embeddings During Training",
    },
    scene: {
      xaxis: axis_config(embeddings, 0),
      yaxis: axis_config(embeddings, 1),
      zaxis: axis_config(embeddings, 2),
      aspectratio: {
        x: 1,
        y: 1,
        z: 1,
      },
    },
    showlegend: false,
  };
}

function make_scatter_trace(embeddings, timesteps) {
  const idx = embeddings.length - 1;
  const data = embeddings[idx];

  return {
    x: data[0],
    y: data[1],
    z: data[2],
    hovertemplate:
      "%{text}<br>" +
      "(%{x:.1f}, %{y:.1f}, %{z:.1f})<br>" +
      "Step: " +
      timesteps[timesteps.length - 1] +
      "<br>" +
      "<extra></extra>",
    hovername: tokens_list,
    text: tokens_list,
    mode: "markers",
    marker: {
      color: colors,
    },
    type: "scatter3d",
  };
}

function make_line_trace(embeddings, timesteps, tokenIdx) {
  return {
    x: embeddings.map((e) => e[0][tokenIdx]),
    y: embeddings.map((e) => e[1][tokenIdx]),
    z: embeddings.map((e) => e[2][tokenIdx]),
    type: "scatter3d",
    mode: "lines",
    hovertemplate:
      tokens_list[tokenIdx] +
      "<br>" +
      "(%{x:.1f}, %{y:.1f}, %{z:.1f})<br>" +
      "Step: %{customdata}" +
      "<br>" +
      "<extra></extra>",

    customdata: timesteps,
    line: {
      width: 4,
      color: colors[tokenIdx],
      reversescale: false,
    },
  };
}

function plot_embeddings_over_time(div, embeddings, timesteps) {
  const scatter_trace = make_scatter_trace(embeddings, timesteps);
  const line_traces = [...Array(n_tokens).keys()].map((tokenIdx) =>
    make_line_trace(embeddings, timesteps, tokenIdx),
  );

  const layout = make_layout(embeddings);

  Plotly.newPlot(div, [scatter_trace, ...line_traces], layout);
}

fetch("embeddings_over_time.json")
  .then((response) => response.json())
  .then((json) => {
    plot_embeddings_over_time(
      "plot-over-time",
      json["embeddings"],
      json["timesteps"],
    );
    plot_embeddings_over_time(
      "plot-over-time",
      json["embeddings"],
      json["timesteps"],
    );
  });
