function sigmoidGradient(sigmoidValue) {
  return sigmoidValue.mul(tf.sub(1, sigmoidValue));
}
function reluGradient(value) {
  return value.greater(0).toFloat();
}

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights_input_hidden = tf.randomNormal(
      [this.hidden_nodes, this.input_nodes],
      0,
      Math.sqrt(2 / (this.input_nodes + this.hidden_nodes))
    );
    this.weights_hidden_output = tf.randomNormal(
      [this.output_nodes, this.hidden_nodes],
      0,
      Math.sqrt(2 / (this.hidden_nodes + this.output_nodes))
    );

    this.bias_hidden = tf.randomUniform([this.hidden_nodes, 1], -1, 1);
    this.bias_output = tf.randomUniform([this.output_nodes, 1], -1, 1);

    this.learningRate = 0.001;
  }

  feedForward(input_array) {
    let inputs = tf.tensor2d(input_array, [input_array.length, 1]);

    let hidden = tf.matMul(this.weights_input_hidden, inputs);
    hidden = tf.add(hidden, this.bias_hidden);
    hidden = tf.relu(hidden);

    let outputs = tf.matMul(this.weights_hidden_output, hidden);
    outputs = tf.add(outputs, this.bias_output);
    outputs = outputs.sigmoid();
    return outputs;
  }

  train(input_array, target_array) {
    let inputs = tf.tensor2d(input_array, [input_array.length, 1]);

    let hidden = tf.matMul(this.weights_input_hidden, inputs);
    hidden = tf.add(hidden, this.bias_hidden);
    hidden = tf.relu(hidden);

    let outputs = tf.matMul(this.weights_hidden_output, hidden);
    outputs = tf.add(outputs, this.bias_output);
    outputs = outputs.sigmoid();

    let targets = tf.tensor2d(target_array, [target_array.length, 1]);

    let output_errors = tf.sub(targets, outputs);
    let gradients = sigmoidGradient(outputs);

    gradients = gradients.mul(output_errors);
    gradients = gradients.mul(this.learningRate);

    let hidden_Transpose = tf.transpose(hidden);
    let weights_hidden_output_Deltas = tf.matMul(gradients, hidden_Transpose);

    this.weights_hidden_output = tf.add(
      this.weights_hidden_output,
      weights_hidden_output_Deltas
    );

    this.bias_output = tf.add(this.bias_output, gradients);

    let weights_hidden_output_Transpose = tf.transpose(
      this.weights_hidden_output
    );
    let hidden_errors = tf.matMul(
      weights_hidden_output_Transpose,
      output_errors
    );

    let hidden_gradient = reluGradient(hidden);

    hidden_gradient = hidden_gradient.mul(hidden_errors);
    hidden_gradient = hidden_gradient.mul(this.learningRate);

    let inputs_Transpose = tf.transpose(inputs);
    let weights_input_hidden_Deltas = tf.matMul(
      hidden_gradient,
      inputs_Transpose
    );

    this.weights_input_hidden = tf.add(
      this.weights_input_hidden,
      weights_input_hidden_Deltas
    );

    this.bias_hidden = tf.add(this.bias_hidden, hidden_gradient);
  }
}

export default NeuralNetwork;
