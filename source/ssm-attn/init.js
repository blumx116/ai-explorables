let dt_vis = d3.select("#dt-visualization");

dt_vis.select("*").remove();

let svg = d3
  .select("#graph")
  .append("svg")
  .style("background-size", "contain")
  .attr("height", "100%")
  .attr("width", "100%");

let bg0 = svg
  .append("image")
  .attr("width", "100%")
  .attr("xlink:href", "img/paper-clip.png")
  .attr("opacity", 0);

let bg1 = svg
  .append("image")
  .attr("height", "100%")
  .attr("width", "100%")
  .attr("xlink:href", "img/layer_only.svg")
  .attr("opacity", 0);

let bg2 = svg
  .append("image")
  .attr("height", "100%")
  .attr("width", "100%")
  .attr("xlink:href", "img/full_model.svg")
  .attr("opacity", 0);

function svg_rectangle({ x, y, height, width }) {
  return svg
    .append("rect")
    .attr("x", x)
    .attr("y", y)
    .attr("height", height)
    .attr("width", width)
    .attr("fill", "transparent")
    .attr("stroke", "#69b3a2")
    .attr("stroke-width", "4px")
    .attr("opacity", 0);
}

let ssm_rect = svg_rectangle({
  x: 80,
  y: 255,
  height: 45,
  width: 76,
});

let conv_rect = svg_rectangle({
  x: 80,
  y: 334,
  height: 45,
  width: 76,
});

let gate_rect = svg_rectangle({
  x: 165,
  y: 230,
  height: 190,
  width: 105,
});

let mul_rect = svg_rectangle({
  x: 103,
  y: 223,
  height: 30,
  width: 30,
});

let obj2section = [
  [[0], bg0, "bg0"],
  [[1, 2, 3], bg1, "bg1"],
  [[4, 5, 6, 7, 8], bg2, "bg2"],
  [[1, 4], ssm_rect, "ssm_rect"],
  [[5], conv_rect, "conv_rect"],
  [[8], gate_rect, "gate_rect"],
  [[8], mul_rect, "mul_rect"],
];

function draw_chart(i) {
  obj2section.forEach((e) => {
    const [sections, rect, name] = e;
    console.log({ sections, rect, name });
    rect.transition().duration(500).attr("opacity", +sections.includes(i));
  });

  d3.select("#graph-title")
    .transition(500)
    .style("opacity", +(i != 0));

  d3.select("#intro-text")
    .transition(500)
    .style("opacity", +(i == 0));
}

function textToIndices(text) {
  var result = [];

  for (var i = 0; i < text.length; i++) {
    result[i] = textTokenMap[text[i].toUpperCase()];
  }
  return result;
}

async function dtProjDataFromIndices(indices) {
  const dt_proj = await window
    .tfjs_model(indices)
    .mamba_layers[0].dt_proj.buffer();

  const result = [];

  for (var i = 0; i < indices.length; i++) {
    result[i] = {
      token: indices[i],
      delta: dt_proj.get(i, 0),
      idx: i,
    };
  }

  return result;
}

async function __input_box_callback() {
  const values = $("#input-box").val();
  const indices = textToIndices(values);

  if (window.tfjs_model) {
    const data = await dtProjDataFromIndices(indices);

    if (window.visualize_dt_timeout) {
      clearTimeout(window.visualize_dt_timeout);
      dt_vis.selectAll("*").remove();
    }
    dt_plot(dt_vis, data);

    const outputs = await window.tfjs_model(indices);
    visualize_weights_plotly(
      "conv-input-heatmap",
      tf.transpose(outputs.mamba_layers[0].raw_hidden_states, [1, 0]),
      "Inputs to Convolution",
      "SSM Series",
      "Token",
      indices,
    );

    visualize_weights_plotly(
      "gate-heatmap",
      outputs.mamba_layers[0].silu_gate.slice([0, 0], [1, -1]),
      "Value of Gate After Activation",
      "SSM Series",
      "Token",
      indices,
    );
  }
}

function input_box_callback() {
  if (window.cb_t) {
    clearTimeout(window.cb_t);
  }
  window.cb_t = setTimeout(__input_box_callback, 1000);
}

function createWeightDropdown(weights) {
  var dropdown = $("#weight-dropdown");
  for (key in weights) {
    $("<option />", { value: key, text: tfjs_weights[key].name }).appendTo(
      dropdown,
    );
  }
}

async function onModelLoad() {
  if (window.model_load_timeout) {
    clearTimeout(window.model_load_timeout);
    window.model_load_timeout = undefined;
  }

  if (window.tfjs_model) {
    input_box_callback();

    visualize_weights_plotly(
      "conv-heatmap",
      window.tfjs_weights.conv.tensor.squeeze(1),
      "Convolution Weights",
      "SSM Series",
      "Kernel (t)",
    );
  } else {
    window.model_load_timeout = setTimeout(onModelLoad, 100);
  }
}

onModelLoad();

d3.graphScroll()
  .sections(d3.selectAll("#sections > div.scroll-section"))
  .on("active", function (i) {
    draw_chart(i);
  });
