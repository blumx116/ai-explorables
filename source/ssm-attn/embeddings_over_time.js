const colors = ["#FAEDCB", "#C9E4DE", "#DBCDF0", "#FD8A8A"];
const n_axes = 3;
const n_tokens = 4;

function make_frame(embeddings, t) {
  const data = embeddings[t];
  return {
    x: data[0],
    y: data[1],
    z: data[2],
    hovername: ["A", "B", "C", "✻"],
    text: ["A", "B", "C", "✻"],
    mode: "markers",
    marker: {
      color: colors,
    },
    hovertemplate:
      "%{text}<br>" + "(%{x:.1f}, %{y:.1f}, %{z:.1f})" + "<extra></extra>",

    name: "Frame " + t,
    type: "scatter3d",
  };
}

function axis_minmax(embeddings, axis) {
  var values = [];
  for (var i = 0; i < embeddings.length; i++) {
    values = [...values, ...embeddings[i][axis]];
  }
  return [Math.min(values), Math.max(values)];
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

function vis_embed_over_time(div, embeddings, timesteps) {
  var frames = [];
  for (var i = 0; i < embeddings.length; i++) {
    frames = [...frames, make_frame(embeddings, i)];
  }

  console.log({ i: frames[frames.length - 1] });

  var layout = {
    title: {
      text: "Embedding of each token over time",
    },
    xaxis: axis_config(embeddings, 0),
    yaxis: axis_config(embeddings, 1),
    zaxis: axis_config(embeddings, 2),
  };

  Plotly.newPlot(div, [frames[frames.length - 1]], layout);
}

fetch("embeddings_over_time.json")
  .then((response) => response.json())
  .then((json) => {
    vis_embed_over_time(
      "plot-over-time",
      json["embeddings"],
      json["timesteps"],
    );
  });
