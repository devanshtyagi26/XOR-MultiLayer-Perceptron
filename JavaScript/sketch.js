import NeuralNetwork from "./neuralNetwork.js";
import { training_data } from "./trainingData.js";

let canvasWidth = 435;
let canvasHeight = 435;
let isCanvasActive = false;
let canvas;
let nn;

// Function to handle form submission
document
  .getElementById("canvasForm")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent page reload

    clearCanvasVariables();

    // Show canvas container and activate canvas
    document.getElementById("canvasContainer").style.display = "flex";

    canvas = createCanvas(canvasWidth, canvasHeight, WEBGL);
    canvas.parent("canvasContainer"); // Attach canvas to #canvasContainer

    canvas.style("background", "grey");
    // canvas.style("border", "0.1px solid rgba(255, 255, 255, 0.36)");
    // canvas.style("box-shadow", "8px 8px 21px black");
    // canvas.style("border-radius", "15px");

    // Remove the <main> tag if it exists
    let mainElement = document.querySelector("main");
    if (mainElement) {
      mainElement.remove(); // This removes the <main> tag
    }

    // Initialize the neural network
    nn = new NeuralNetwork(2, 8, 1);

    isCanvasActive = true;
  });

window.setup = function () {
  document.getElementById("canvasContainer").style.display = "none";
};

// window.draw = function () {
  //   if (!isCanvasActive) {
    //     return; // Don't execute draw until the form is submitted
    //   }
    //   for (let i = 0; i < 500; i++) {
      //     let data = random(training_data); // Correct way to pick random data
      //     nn.train(data.inputs, data.targets);
      //   }
      
      //   nn.setLearningRate(0.0001);
      //   let resolution = 5;
      //   let cols = width / resolution;
      //   let rows = height / resolution;
      //   for (let i = 0; i < cols; i++) {
//     for (let j = 0; j < rows; j++) {
  //       let p = i / cols;
//       let q = j / rows;
//       let inputs = [p, q];
//       let y = nn.feedForward(inputs);
//       noStroke();
//       fill(y * 255);
//       rect(i * resolution, j * resolution, resolution, resolution);
//     }
//   }
// };

// 3D
  window.draw = function () {
  if (!isCanvasActive) {
    return; // Stop execution if the canvas is not active
  }

  background(30);
  rotateX(PI / 3); // Rotate to get a better 3D perspective
  translate(-width / 2, -height / 2);

  // Train the neural network with random samples
  for (let i = 0; i < 500; i++) {
    let data = random(training_data); // Replace with your dataset
    nn.train(data.inputs, data.targets);
  }

  nn.setLearningRate(0.0001);
  let resolution = 5;
    let cols = width / resolution;
    let rows = height / resolution;
  // Draw 3D graph based on neural network outputs
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let p = i / cols;
      let q = j / rows;
      let inputs = [p, q];
      let y = nn.feedForward(inputs); // Neural network prediction

      // Map the output to height
      let h = map(y, 0, 1, -150, 150);

      // Color based on output
      let col = color(map(y, 0, 1, 0, 255), 150, 255);

      // Draw box for each point
      push();
      translate(i * resolution, j * resolution, h / 2); // Adjust position
      fill(col);
      noStroke();
      box(resolution, resolution, h); // Create 3D box
      pop();
    }
  }
}

// Function to clear all variables
function clearCanvasVariables() {
  isCanvasActive = false;
  // Hide the canvas container
  document.getElementById("canvasContainer").style.display = "none";

  // Remove any existing canvas
  if (canvas) {
    canvas.remove();
  }
  loop();
}

document.getElementById("resetButton").addEventListener("click", function () {
  resetEverything();
});

function resetEverything() {
  // Clear previous canvas and points
  clearCanvasVariables();

  // Optionally reset form values or leave them as is
  document.getElementById("canvasForm").reset(); // This will reset the form inputs if needed

  // Disable the canvas
  isCanvasActive = false;

  // Optionally hide the canvas container again
  document.getElementById("canvasContainer").style.display = "block";

  // Log or alert that everything has been reset
  console.log("Everything has been reset.");
}
