import NeuralNetwork from "./neuralNetwork.js";
import { training_data } from "./trainingData.js";

let canvasWidth = 435;
let canvasHeight = 435;
let isCanvasActive = false;
let canvas;
// Create a new instance of the Neural Network
let nn = new NeuralNetwork(2, 2, 1);

// Function to get a random data entry from the training data
function random(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// Train the neural network with random data from training_data
for (let i = 0; i < 2000; i++) {
  let data = random(training_data);
  nn.train(data.inputs, data.targets);
}

// Test the neural network with different inputs
console.log(nn.feedForward([0, 0]).arraySync()); // Expected output: close to 0
console.log(nn.feedForward([1, 1]).arraySync()); // Expected output: close to 0
console.log(nn.feedForward([1, 0]).arraySync()); // Expected output: close to 1
console.log(nn.feedForward([0, 1]).arraySync()); // Expected output: close to 1

// // Function to handle form submission
// document
//   .getElementById("canvasForm")
//   .addEventListener("submit", function (event) {
//     event.preventDefault(); // Prevent page reload

//     clearCanvasVariables();

//     // Show canvas container and activate canvas
//     document.getElementById("canvasContainer").style.display = "flex";

//     canvas = createCanvas(canvasWidth, canvasHeight);
//     canvas.parent("canvasContainer"); // Attach canvas to #canvasContainer

//     canvas.style("background", "grey");
//     // canvas.style("border", "0.1px solid rgba(255, 255, 255, 0.36)");
//     // canvas.style("box-shadow", "8px 8px 21px black");
//     // canvas.style("border-radius", "15px");

//     // Remove the <main> tag if it exists
//     let mainElement = document.querySelector("main");
//     if (mainElement) {
//       mainElement.remove(); // This removes the <main> tag
//     }

//     // Initialize the neural network
//     nn = new NeuralNetwork(2, 2, 1);

//     // Train the neural network with random data from training_data
//     for (let i = 0; i < 5000; i++) {
//       let data = random(training_data); // Correct way to pick random data
//       nn.train(data.inputs, data.targets);
//     }

//     // Test the neural network with different inputs
//     console.log(nn.feedForward([0, 0]).arraySync());
//     console.log(nn.feedForward([1, 1]).arraySync());
//     console.log(nn.feedForward([1, 0]).arraySync());
//     console.log(nn.feedForward([0, 1]).arraySync());

//     isCanvasActive = true;
//   });

// window.setup = function () {
//   document.getElementById("canvasContainer").style.display = "none";
// };

// window.draw = function () {
//   if (!isCanvasActive) {
//     return; // Don't execute draw until the form is submitted
//   }
// };

// // Function to clear all variables
// function clearCanvasVariables() {
//   isCanvasActive = false;
//   // Hide the canvas container
//   document.getElementById("canvasContainer").style.display = "none";

//   // Remove any existing canvas
//   if (canvas) {
//     canvas.remove();
//   }
//   loop();
// }

// document.getElementById("resetButton").addEventListener("click", function () {
//   resetEverything();
// });

// function resetEverything() {
//   // Clear previous canvas and points
//   clearCanvasVariables();

//   // Optionally reset form values or leave them as is
//   document.getElementById("canvasForm").reset(); // This will reset the form inputs if needed

//   // Disable the canvas
//   isCanvasActive = false;

//   // Optionally hide the canvas container again
//   document.getElementById("canvasContainer").style.display = "block";

//   // Log or alert that everything has been reset
//   console.log("Everything has been reset.");
// }
