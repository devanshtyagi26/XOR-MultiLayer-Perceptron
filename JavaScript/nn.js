import NeuralNetwork from "./neuralNetwork.js";
import { training_data } from "./trainingData.js";

function pickRandomObject(array) {
  const randomIndex = Math.floor(Math.random() * array.length);
  return array[randomIndex];
}

let nn = new NeuralNetwork(2, 20, 1);
for (let i = 0; i < 5000; i++) {
  nn.learningRate = 0.1 / (1 + 0.001 * i); // Example decay schedule

  let data = pickRandomObject(training_data);
  nn.train(data.inputs, data.targets);
  nn.feedForward([0, 0]).print();
  nn.feedForward([1, 1]).print();
  nn.feedForward([1, 0]).print();
  nn.feedForward([0, 1]).print();
}
