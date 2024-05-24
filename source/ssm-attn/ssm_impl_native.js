const vocab_size = 4; // v
const dt_rank = 1;
const ssm_state_size = 4; // d

function p(name, tensor) {
  console.log(name);
  console.log(tensor.shape);
  tensor.print();
  for (var i = 0; i < 3; i++) {
    console.log("============================================");
  }
}

function grouped_conv(weights, x) {
  const conv_weight = weights.weights.squeeze().transpose([1, 0]);
  const [s, h] = x.shape;
  const k = weights.weights.shape[2];
  var output_tensors = [];
  const padded_x = tf.concat([tf.zeros([k - 1, h]), x], 0);

  for (var i = 0; i < s; i++) {
    var conv_patch = padded_x.slice([i, 0], [k, -1]);
    var conv_values = tf.mul(conv_patch, conv_weight).sum(0).expandDims(0);
    conv_values = tf.add(conv_values, weights.bias);
    output_tensors = [...output_tensors, conv_values];
  }

  return tf.concat(output_tensors, 0);
}

function SSM(A, B, C, D, u) {
  const [h, s, d] = A.shape;
  var ssm_state = tf.zeros([h, d]);
  var ssm_states = [];
  var scan_outputs = [];
  for (var i = 0; i < s; i++) {
    const Ai = A.slice([0, i, 0], [-1, 1, -1]).squeeze([1]);
    const Bi = B.slice([0, i, 0], [-1, 1, -1]).squeeze([1]);
    const ui = u.slice([i, 0], [1, -1]).squeeze([0]).expandDims(-1);
    const Ci = C.slice([i, 0], [1, -1]).squeeze([0]).expandDims(-1);

    ssm_state = tf.add(tf.mul(Ai, ssm_state), tf.mul(Bi, ui));
    ssm_states = [...ssm_states, ssm_state.expandDims(0)];

    const scan_output = tf.add(
      tf.matMul(ssm_state, Ci),
      tf.mul(ui, D.expandDims(-1)),
    );

    scan_outputs = [...scan_outputs, scan_output];
  }

  const all_ssm_states = tf.concat(ssm_states, 1);
  const all_scan_outputs = tf.concat(scan_outputs, -1);
  return {
    all_ssm_states,
    all_scan_outputs,
  };
}

function silu(x) {
  return tf.mul(x, tf.sigmoid(x));
}

function mamba_layer(weights, x) {
  const projected_states = tf
    .matMul(x, weights.in_proj_weight.transpose([1, 0]))
    .transpose([1, 0]);

  var [raw_hidden_states, gate] = tf.split(projected_states, 2, 0);
  raw_hidden_states = raw_hidden_states.transpose([1, 0]);
  p("raw_hidden_states", raw_hidden_states);
  p("gate", gate);

  var hidden_states = silu(grouped_conv(weights.conv, raw_hidden_states));

  var ssm_parameters = tf.matMul(
    hidden_states,
    weights.x_proj_weight.transpose([1, 0]),
  );

  var [timestep, B, C] = tf.split(
    ssm_parameters,
    [dt_rank, ssm_state_size, ssm_state_size],
    -1,
  );

  var discrete_time_step = tf.add(
    tf.matMul(timestep, weights.dt_proj_weight.transpose([1, 0])),
    weights.dt_proj_bias,
  );

  discrete_time_step = tf.softplus(discrete_time_step).transpose([1, 0]);

  const A = tf.mul(tf.exp(weights.A_log), -1);
  const discrete_A = tf.exp(
    tf.mul(A.expandDims(1), discrete_time_step.expandDims(-1)),
  );

  const discrete_B = tf.mul(B.expandDims(0), discrete_time_step.expandDims(-1));

  const ssm_output = SSM(discrete_A, discrete_B, C, weights.D, hidden_states);
  const gated_scan_outputs = tf.mul(ssm_output.all_scan_outputs, silu(gate));

  const result = tf.matMul(
    gated_scan_outputs.transpose([1, 0]),
    weights.out_proj_weight.transpose([1, 0]),
  );

  return {
    output: result,
    dt_proj: timestep,
    A: A,
    discrete_A: discrete_A,
    B: B,
    discrete_B: discrete_B,
    gated_scan_outputs: gated_scan_outputs,
    projected_states: projected_states,
    raw_hidden_states: raw_hidden_states,
    hidden_states: hidden_states,
  };
}

function model(weights, x) {
  var repr = tf.oneHot(x.toInt(), vocab_size);
  var repr = tf.matMul(repr, weights.embedding_weight);
  const results = [];

  for (var i = 0; i < weights.layers.length; i++) {
    var mambaLayerOutput = mamba_layer(weights.layers[i], repr);
    results[i] = mambaLayerOutput;
    repr = tf.add(repr, mambaLayerOutput.output);
  }

  const output = tf.matMul(repr, weights.embedding_weight.transpose([1, 0]));

  return {
    output: output,
    embeddings: repr,
    mamba_layers: results,
  };
}

function json2MambaWeights(json, layerIdx) {
  const layer = "backbone.layers." + layerIdx.toString() + ".mixer.";

  function load(name) {
    return tf.tensor(json[layer + name]);
  }

  return {
    in_proj_weight: load("in_proj.weight"),
    x_proj_weight: load("x_proj.weight"),
    dt_proj_weight: load("dt_proj.weight"),
    dt_proj_bias: load("dt_proj.bias"),
    A_log: load("A_log"),
    D: load("D"),
    conv: {
      weights: load("conv1d.weight"),
      bias: load("conv1d.bias"),
    },
    out_proj_weight: load("out_proj.weight"),
  };
}

function json2Weights(json) {
  var layers = [];
  var layerIdx = 0;
  while (true) {
    try {
      layers = [...layers, json2MambaWeights(json, layerIdx)];
      layerIdx += 1;
    } catch {
      break;
    }
  }
  return {
    embedding_weight: tf.tensor(json["backbone.embeddings.weight"]),
    layers: layers,
  };
}

function weightsWithNames(weights) {
  const [hidden_eff, gate_eff] = tf.split(
    tf.matMul(
      weights.embedding_weight,
      weights.layers[0].in_proj_weight.transpose([1, 0]),
    ),
    2,
    -1,
  );
  return {
    hidden_eff: {
      name: "Hidden States Embedding",
      tensor: hidden_eff,
    },
    gate_eff: {
      name: "Gate Circuit",
      tensor: gate_eff,
    },
    conv: {
      name: "1D Convolution Weights",
      tensor: weights.layers[0].conv.weights,
    },
    A: {
      name: "A",
      tensor: tf.mul(tf.exp(weights.layers[0].A_log), -1),
    },
  };
}

// Load the JSON weights
fetch("model.json")
  .then((response) => response.json())
  .then((json) => {
    const weights = json2Weights(json);
    console.log(weights);
    window.tfjs_model = (arr) => {
      return model(weights, tf.tensor(arr));
    };
    window.tfjs_weights = weightsWithNames(weights);
  });
