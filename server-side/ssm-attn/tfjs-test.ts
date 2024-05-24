import "@tensorflow/tfjs-backend-webgl";

import * as tfconv from "@tensorflow/tfjs-converter";
import * as tf from "@tensorflow/tfjs-core";
import * as tfnode from "@tensorflow/tfjs-node";

(async () => {
  tf.ENV.set("KEEP_INTERMEDIATE_TENSORS", true);
  const handler = tfnode.io.fileSystem(
    "/Users/carterblum/projects/pair/ai-explorables/server-side/ssm-attn/deleteme/model.json",
  );
  const model = await tfconv.loadGraphModel(handler);
  const tokens = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const result = model.execute(tokens);
  const tensors = model.getIntermediateTensors();

  console.log(result);
  console.log(tensors);
})();
