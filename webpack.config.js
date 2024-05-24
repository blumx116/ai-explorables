const path = require("path");

module.exports = {
  entry: "./source/ssm-attn/init.ts",
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
  output: {
    filename: "init.js",
    path: path.resolve(__dirname, "source/ssm-attn/"),
  },
};
