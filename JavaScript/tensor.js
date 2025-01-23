

function sigmoid(x) {
  return tf.sigmoid(x);
}

function sigmoidGradient(sigmoidValue) {
  return sigmoidValue.mul(tf.scalar(1).sub(sigmoidValue));
}

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    // Initialize weights and biases as tensors
    this.weights_input_hidden = tf.randomUniform([this.hidden_nodes, this.input_nodes], -1, 1);
    this.weights_hidden_output = tf.randomUniform([this.output_nodes, this.hidden_nodes], -1, 1);

    this.bias_hidden = tf.randomUniform([this.hidden_nodes, 1], -1, 1);
    this.bias_output = tf.randomUniform([this.output_nodes, 1], -1, 1);

    this.learningRate = 0.1;
  }

  feedForward(input_array) {
    return tf.tidy(() => {
      // Convert input array to tensor
      const inputs = tf.tensor2d(input_array, [this.input_nodes, 1]);

      // Calculate hidden layer activations
      const hidden = sigmoid(tf.add(tf.matMul(this.weights_input_hidden, inputs), this.bias_hidden));

      // Calculate output layer activations
      const outputs = sigmoid(tf.add(tf.matMul(this.weights_hidden_output, hidden), this.bias_output));

      // Return outputs as array
      return outputs.arraySync();
    });
  }

  train(input_array, target_array) {
    tf.tidy(() => {
      // Convert inputs and targets to tensors
      const inputs = tf.tensor2d(input_array, [this.input_nodes, 1]);
      const targets = tf.tensor2d(target_array, [this.output_nodes, 1]);

      // Forward pass
      const hidden = sigmoid(tf.add(tf.matMul(this.weights_input_hidden, inputs), this.bias_hidden));
      const outputs = sigmoid(tf.add(tf.matMul(this.weights_hidden_output, hidden), this.bias_output));

      // Compute output errors
      const output_errors = targets.sub(outputs);

      // Compute gradients for output layer
      const gradients = sigmoidGradient(outputs).mul(output_errors).mul(this.learningRate);
      const weights_hidden_output_deltas = tf.matMul(gradients, hidden.transpose());

      // Update weights and biases for output layer
      this.weights_hidden_output = this.weights_hidden_output.add(weights_hidden_output_deltas);
      this.bias_output = this.bias_output.add(gradients);

      // Compute hidden layer errors
      const hidden_errors = tf.matMul(this.weights_hidden_output.transpose(), output_errors);

      // Compute gradients for hidden layer
      const hidden_gradients = sigmoidGradient(hidden).mul(hidden_errors).mul(this.learningRate);
      const weights_input_hidden_deltas = tf.matMul(hidden_gradients, inputs.transpose());

      // Update weights and biases for hidden layer
      this.weights_input_hidden = this.weights_input_hidden.add(weights_input_hidden_deltas);
      this.bias_hidden = this.bias_hidden.add(hidden_gradients);
    });
  }

  setLearningRate(learning_rate = 0.1) {
    this.learningRate = learning_rate;
  }
}

export default NeuralNetwork;
