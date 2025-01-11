import NeuralNetwork from "./neuralNetwork.js";

let nn = new NeuralNetwork(2, 2, 1);
let input = [1, 0];

let output = nn.feedForward(input);
output.print();