let dt_vis = d3.select("#app");

dt_vis.select("*").remove();

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
  console.log("called __input_box_callback with " + values);
  const indices = textToIndices(values);
  console.log(indices);

  if (window.tfjs_model) {
    const data = await dtProjDataFromIndices(indices);
    console.log(data);

    if (window.visualize_dt_timeout) {
      clearTimeout(window.visualize_dt_timeout);
      console.log("nuking");
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
  console.log("creating dropdown");
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

    console.log(window.tfjs_weights.layers);
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
