import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";

const vocab_size: number = 4; // v
const dt_rank: number = 1;
const ssm_state_size: number = 4; // d

interface MambaConvWeights {
  weights: tf.Tensor; // (h, 1, k)
  bias: tf.Tensor; // (h, )
}

interface MambaWeights {
  in_proj_weight: tf.Tensor; // (2 * h, D)
  x_proj_weight: tf.Tensor; // (dt_rank + 2d, h)
  dt_proj_weight: tf.Tensor; // (h, dt_rank)
  dt_proj_bias: tf.Tensor; // (h, )
  conv: MambaConvWeights;
  A_log: tf.Tensor; // (h, d)
  D: tf.Tensor; // (h)
  out_proj_weight: tf.Tensor; // (D, h)
}

interface Weights {
  embedding_weight: tf.Tensor; // (D, v)
  layers: MambaWeights[];
}

interface MambaOutputType {
  output: tf.Tensor;
}

interface OutputType {
  output: tf.Tensor; // whatever I feel like for now
  states: MambaOutputType[];
}

function p(name: string, tensor: tf.Tensor): void {
  console.log(name);
  console.log(tensor.shape);
  tensor.print();
  for (var i = 0; i < 3; i++) {
    console.log("============================================");
  }
}

function grouped_conv(weights: MambaConvWeights, x: tf.Tensor): tf.Tensor {
  /**
   * weights: weights for convolution
   * x: (b, s, h)
   */
  const conv_weight: tf.Tensor = weights.weights.squeeze().transpose([1, 0]);
  // conv_weight: (k, h)
  const [b, s, h] = x.shape;
  const k: number = weights.weights.shape[2] as number;
  var output_tensors: tf.Tensor[] = [];
  const padded_x: tf.Tensor = tf.concat([tf.zeros([b, k - 1, h]), x], 1);
  // padded_x: (b, s + k -1, h)

  for (var i: number = 0; i < s; i++) {
    var conv_patch: tf.Tensor = padded_x.slice([0, i, 0], [-1, k, -1]);
    // conv_patch: (b, k, h)
    var conv_values: tf.Tensor = tf
      .mul(conv_patch, conv_weight)
      .sum(1)
      .expandDims(1);
    conv_values = tf.add(conv_values, weights.bias);
    // conv_patch: (b, 1, h)
    output_tensors = [...output_tensors, conv_values];
  }

  return tf.concat(output_tensors, 1);
}

function SSM(
  A: tf.Tensor,
  B: tf.Tensor,
  C: tf.Tensor,
  D: tf.Tensor,
  u: tf.Tensor,
): tf.Tensor {
  /**
   * @param A: (b, h, s, d)
   * @param B: (b, h, s, d)
   * @param C: (b, s, d)
   * @param D: (h,)
   * @param u: (b, s, h)
   *
   * @returns ssm_outputs: (b, h, s)
   */
  const [b, h, s, d] = A.shape;
  var ssm_state: tf.Tensor = tf.zeros([b, h, d]);
  var ssm_states: tf.Tensor[] = []; // each (b, 1, h, d)
  var scan_outputs: tf.Tensor[] = []; // each(b, h, 1)
  for (var i: number = 0; i < s; i++) {
    const Ai = A.slice([0, 0, i, 0], [-1, -1, 1, -1]).squeeze([2]); // (b, h, d)
    const Bi = B.slice([0, 0, i, 0], [-1, -1, 1, -1]).squeeze([2]); // (b, h, d)
    const ui = u.slice([0, i, 0], [-1, 1, -1]).squeeze([1]).expandDims(-1); // (b, h, 1)
    const Ci = C.slice([0, i, 0], [-1, 1, -1]).squeeze([1]).expandDims(-1); // (b, d, 1)

    ssm_state = tf.add(tf.mul(Ai, ssm_state), tf.mul(Bi, ui));
    ssm_states = [...ssm_states, ssm_state.expandDims(1)];

    const scan_output: tf.Tensor = tf.add(
      tf.matMul(ssm_state, Ci), // (b, h, 1),
      tf.mul(ui, D.expandDims(0).expandDims(-1)),
    );

    scan_outputs = [...scan_outputs, scan_output];
  }

  const all_ssm_states: tf.Tensor = tf.concat(ssm_states, 1);
  // all_ssm_states: (b, s, h, d)
  const all_scan_outputs: tf.Tensor = tf.concat(scan_outputs, -1);
  // all_scan_outputs: (b, h, s)
  return all_scan_outputs;
}

function silu(x: tf.Tensor): tf.Tensor {
  return tf.mul(x, tf.sigmoid(x));
}

function mamba_layer(weights: MambaWeights, x: tf.Tensor): MambaOutputType {
  const projected_states: tf.Tensor = tf
    .matMul(x, weights.in_proj_weight.transpose([1, 0]))
    .transpose([0, 2, 1]);
  // projected_states: (b, 2h, s)

  var [raw_hidden_states, gate] = tf.split(projected_states, 2, 1);
  raw_hidden_states = raw_hidden_states.transpose([0, 2, 1]);
  // gate: (b, h, s)
  // raw_hidden_states: (b, s, h)

  var hidden_states: tf.Tensor = silu(
    grouped_conv(weights.conv, raw_hidden_states),
  );
  // hidden_size: (b, s, h)

  var ssm_parameters: tf.Tensor = tf.matMul(
    hidden_states,
    weights.x_proj_weight.transpose([1, 0]),
  );
  // ssm_parameters: (b, s, dt_rank + 2d)

  var [timestep, B, C] = tf.split(
    ssm_parameters,
    [dt_rank, ssm_state_size, ssm_state_size],
    -1,
  );
  // B, C: (b, s, d)
  // timestep: (b, s, dt_rank)

  var discrete_time_step: tf.Tensor = tf.add(
    tf.matMul(timestep, weights.dt_proj_weight.transpose([1, 0])),
    weights.dt_proj_bias,
  );

  discrete_time_step = tf.softplus(discrete_time_step).transpose([0, 2, 1]);
  // discrete_time_step: (b, h, s)

  const A: tf.Tensor = tf.mul(tf.exp(weights.A_log), -1);
  // A: (h, d)
  const discrete_A: tf.Tensor = tf.exp(
    tf.mul(A.expandDims(0).expandDims(2), discrete_time_step.expandDims(-1)),
  );
  // discrete_A: (b, h, s, d)
  p("discrete_A", discrete_A);

  const discrete_B: tf.Tensor = tf.mul(
    B.expandDims(1),
    discrete_time_step.expandDims(-1),
  );
  p("discrete_B", discrete_B);
  // discrete_B: (b, h, s, d)

  const scan_outputs: tf.Tensor = SSM(
    discrete_A,
    discrete_B,
    C,
    weights.D,
    hidden_states,
  );
  // scan_outputs: (b, h, s)
  const gated_scan_outputs: tf.Tensor = tf.mul(scan_outputs, silu(gate));
  // gated_scan_outputs: (b, h, s)

  const result: tf.Tensor = tf.matMul(
    gated_scan_outputs.transpose([0, 2, 1]),
    weights.out_proj_weight.transpose([1, 0]),
  );
  // result: (b, s, D)

  return {
    output: result,
  };
}

function model(weights: Weights, x: tf.Tensor): OutputType {
  /**
   * @param weights: weights for the whole model
   * @param x: (b, s)
   */
  var repr: tf.Tensor = tf.oneHot(x.toInt(), vocab_size);
  var repr: tf.Tensor = tf.matMul(repr, weights.embedding_weight); // (b, s, D)
  const results: MambaOutputType[] = [];

  for (var i = 0; i < weights.layers.length; i++) {
    var mambaLayerOutput: MambaOutputType = mamba_layer(
      weights.layers[i],
      repr,
    );
    results[i] = mambaLayerOutput;
    repr = tf.add(repr, mambaLayerOutput.output);
  }

  const output: tf.Tensor = tf.matMul(
    repr,
    weights.embedding_weight.transpose([1, 0]),
  );

  return {
    output: output,
    states: results,
  };
}

function readJson(path: string): any {
  return JSON.parse(fs.readFileSync(path, "utf8"));
}

function json2MambaWeights(json: any, layerIdx: number): MambaWeights {
  const layer: string = "backbone.layers." + layerIdx.toString() + ".mixer.";

  function load(name: string): tf.Tensor {
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

function json2Weights(json: any): Weights {
  var layers: MambaWeights[] = [];
  var layerIdx: number = 0;
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

const weights: Weights = json2Weights(readJson("model.json"));
const result = model(weights, tf.tensor([[3, 1, 0, 0, 2, 2, 1, 3, 2]]));
p("logits", result.output);
