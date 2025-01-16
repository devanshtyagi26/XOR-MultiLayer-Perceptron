import NeuralNetwork from "./neuralNetwork.js";
import { training_data } from "./trainingData.js";

let canvasWidth = 435;
let canvasHeight = 435;
let isCanvasActive = false;
let canvas;
let nn;
let render = "";

// Function to handle form submission
document
  .getElementById("canvasForm")
  .addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent page reload

    clearCanvasVariables();

    document.getElementById("submit").style.visibility = "hidden";

    // Show canvas container and activate canvas
    document.getElementById("canvasContainer").style.display = "flex";
    var ele = document.getElementsByName("radio");
    for (let i = 0; i < ele.length; i++) {
      if (ele[i].checked) render = ele[i].value;
    }

    if (render === "2D") {
      canvas = createCanvas(canvasWidth, canvasHeight);
    } else if (render === "3D") {
      canvas = createCanvas(canvasWidth, canvasHeight, WEBGL);
    }
    canvas.parent("canvasContainer"); // Attach canvas to #canvasContainer

    canvas.style("background", "30");

    // Remove the <main> tag if it exists
    let mainElement = document.querySelector("main");
    if (mainElement) {
      mainElement.remove(); // This removes the <main> tag
    }

    // Initialize the neural network
    nn = new NeuralNetwork(2, 8, 1);
    nn.setLearningRate(0.001);

    isCanvasActive = true;
  });

window.setup = function () {
  document.getElementById("canvasContainer").style.display = "none";
};

// 3D
window.draw = function () {
  if (!isCanvasActive) {
    return; // Stop execution if the canvas is not active
  }
  if (render === "2D") {
    two_DRender();
  } else if (render === "3D") {
    three_DRender();
  }
};

function two_DRender() {
  for (let i = 0; i < 500; i++) {
    let data = random(training_data); // Correct way to pick random data
    nn.train(data.inputs, data.targets);
  }

  let resolution = 5;
  let cols = width / resolution;
  let rows = height / resolution;
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let p = i / cols;
      let q = j / rows;
      let inputs = [p, q];
      let y = nn.feedForward(inputs);
      noStroke();
      fill(y * 255);
      rectMode(CENTER);
      rect(i * resolution, j * resolution, resolution, resolution);
    }
  }
}

function three_DRender() {
  background(30);
  let rotationAngleZ = frameCount * 0.01;

  // Rotate for a better 3D perspective (optional)
  rotateX(PI / 4);
  rotateZ(rotationAngleZ);

  // Train the neural network with random samples
  for (let i = 0; i < 500; i++) {
    let data = random(training_data); // Replace with your dataset
    nn.train(data.inputs, data.targets);
  }

  let resolution = 4;
  let cols = width / resolution - 40;
  let rows = height / resolution - 40;

  // Center the graph by adjusting the translation
  let xOffset = -(cols * resolution) / 2;
  let yOffset = -(rows * resolution) / 2;

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
      let col = color(255, map(y, 0, 1, 0, 255), 0);

      // Draw box for each point
      push();
      translate(xOffset + i * resolution, yOffset + j * resolution, h / 2); // Adjust position
      fill(col);
      noStroke();
      box(resolution, resolution, 5); // Create 3D box
      pop();

      // Synchronize rotation angles

      // Draw the cube in 3D at the center of the canvas
      push();
      noFill(); // No fill for the cube
      stroke(255); // Black stroke
      strokeWeight(1); // Stroke thickness
      // rotateX(PI / 4);
      rotateZ(rotationAngleZ);
      box(270); // Cube with size 100
      pop();
    }
  }
}

// Function to clear all variables
function clearCanvasVariables() {
  isCanvasActive = false;
  // Hide the canvas container
  document.getElementById("canvasContainer").style.display = "none";
  render = "";

  // Remove any existing canvas
  if (canvas) {
    canvas.remove();
  }
  loop();
}

let resetInProgress = false; // Flag to check if reset is in progress

document
  .getElementById("resetButton")
  .addEventListener("click", function (event) {
    event.preventDefault(); // Prevent form submission when reset is clicked

    if (resetInProgress) return; // Prevent reset if it's already in progress

    resetEverything(); // Call the reset function
  });

function resetEverything() {
  resetInProgress = true; // Set the flag to true to indicate reset is happening

  // Clear previous canvas and points
  clearCanvasVariables();

  // Optionally reset form values or leave them as is
  document.getElementById("canvasForm").reset(); // This will reset the form inputs if needed

  // Disable the canvas
  isCanvasActive = false;

  // Optionally hide the canvas container again
  document.getElementById("canvasContainer").style.display = "block";

  document.getElementById("submit").style.visibility = "visible";

  // Log or alert that everything has been reset

  resetInProgress = false; // Set the flag to true to indicate reset is happening
}
