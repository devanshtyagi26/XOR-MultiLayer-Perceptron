import Matrix from "./matrixMath.js";

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function sigmoidGradient(sigmoidValue) {
  return sigmoidValue * (1 - sigmoidValue);
}

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights_input_hidden = new Matrix(this.hidden_nodes, this.input_nodes);

    this.weights_hidden_output = new Matrix(
      this.output_nodes,
      this.hidden_nodes
    );

    this.weights_input_hidden.randomize();
    this.weights_hidden_output.randomize();

    this.bias_hidden = new Matrix(this.hidden_nodes, 1);
    this.bias_output = new Matrix(output_nodes, 1);

    this.bias_hidden.randomize();
    this.bias_output.randomize();

    this.learningRate = 0.1;
  }

  feedForward(input_array) {
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_input_hidden, inputs);
    hidden.add(this.bias_hidden);
    hidden.map(sigmoid);

    let outputs = Matrix.multiply(this.weights_hidden_output, hidden);

    outputs.add(this.bias_output);
    outputs.map(sigmoid);
    return outputs.toArray();
  }

  train(input_array, target_array) {
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_input_hidden, inputs);
    hidden.add(this.bias_hidden);
    hidden.map(sigmoid);

    let outputs = Matrix.multiply(this.weights_hidden_output, hidden);

    outputs.add(this.bias_output);
    outputs.map(sigmoid);

    let targets = Matrix.fromArray(target_array);
    let output_errors = Matrix.subtract(targets, outputs);

    let gradients = Matrix.map(outputs, sigmoidGradient);

    gradients.multiply(output_errors);
    gradients.multiply(this.learningRate);

    let hidden_Transpose = Matrix.transpose(hidden);
    let weights_hidden_output_Deltas = Matrix.multiply(
      gradients,
      hidden_Transpose
    );

    this.weights_hidden_output.add(weights_hidden_output_Deltas);

    this.bias_output.add(gradients);

    let weights_hidden_output_Transpose = Matrix.transpose(
      this.weights_hidden_output
    );
    let hidden_errors = Matrix.multiply(
      weights_hidden_output_Transpose,
      output_errors
    );

    let hidden_gradient = Matrix.map(hidden, sigmoidGradient);

    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learningRate);

    let inputs_Transpose = Matrix.transpose(inputs);
    let weights_input_hidden_Deltas = Matrix.multiply(
      hidden_gradient,
      inputs_Transpose
    );

    this.weights_input_hidden.add(weights_input_hidden_Deltas);
    this.bias_hidden.add(hidden_gradient);
  }
}

export default NeuralNetwork;
