function add_token_ticks(layout, tokens) {
  const tokenChars = [];
  for (var i = 0; i < tokens.length; i++) {
    tokenChars[i] = tokenTextMap[tokens[i]];
  }
  layout.ticktext = tokenChars;
}

function convert_colorscale(min, max, cmap) {
  const range = linspace(min, max, cmap.length);
  const result = [];

  for (var i = 0; i < cmap.length; i++) {
    result[i] = [range[i], cmap[i]];
  }

  return result;
}

async function visualize_weights_plotly(
  div,
  tensor,
  title,
  ytitle,
  xtitle,
  tokens,
) {
  const [ydim, xdim] = tensor.shape;
  const converted_colorscale = convert_colorscale(
    await tensor.min().array(),
    await tensor.max().array(),
    cmap_range,
  );

  var data = {
    z: await tensor.array(),
    type: "heatmap",
    colorscale: "RdBu",
    showscale: false,
    hovertemplate:
      "Value: %{z:.2f}<br>" +
      ytitle +
      ": %{y}<br>" +
      xtitle +
      ": %{x}<extra></extra>",
  };

  var layout = {
    title: {
      text: title,
    },
    yaxis: {
      tickvals: linspace(0, ydim - 1, ydim),
      ticktext: linspace(0, ydim - 1, ydim),
      title: {
        text: ytitle,
      },
      constrain: "domain",
    },
    xaxis: {
      tickvals: linspace(0, xdim - 1, xdim),
      ticktext: linspace(0, xdim - 1, xdim),
      title: {
        text: xtitle,
      },
      scaleanchor: "y",
      constrain: "domain",
    },
  };

  if (tokens) {
    if (ytitle == "Token") {
      add_token_ticks(layout.yaxis, tokens);
    } else if (xtitle == "Token") {
      add_token_ticks(layout.xaxis, tokens);
    }
  }

  Plotly.newPlot(div, [data], layout);
}
