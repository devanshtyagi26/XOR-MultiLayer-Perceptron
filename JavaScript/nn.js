import NeuralNetwork from "./neuralNetwork.js";
import { training_data } from "./trainingData.js";

new p5();  // you need to call this fn first

function setup() {
  let nn = new NeuralNetwork(2, 8, 1);

  for (let i = 0; i < 500000; i++) {
    let data = random(training_data);
    nn.train(data.inputs, data.targets);
  }

  console.table(nn.feedForward([0, 0]));
  console.table(nn.feedForward([1, 1]));
  console.table(nn.feedForward([1, 0]));
  console.table(nn.feedForward([0, 1]));
}

setup();
