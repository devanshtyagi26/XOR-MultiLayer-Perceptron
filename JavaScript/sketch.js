import NeuralNetwork from "./neuralNetwork.js";
import { training_data } from "./trainingData.js";

let canvasWidth = 535;
let canvasHeight = 535;
let isCanvasActive = false;
let canvas;
let nn = null;
let render = "";

  document.getElementById("submit").addEventListener("click", function (event) {
    event.preventDefault(); // Prevent page reload

  clearCanvasVariables();

  document.getElementById("submit").style.visibility = "hidden";

  // Show canvas container and activate canvas
  document.getElementById("canvasContainer").style.display = "flex";
  var ele = document.getElementsByName("radio");
  for (let i = 0; i < ele.length; i++) {
    if (ele[i].checked) render = ele[i].value;
  }
  let summary = document.querySelector(".Summary");
  summary.style.display = "flex";
  if (render === "2D") {
    canvas = createCanvas(canvasWidth, canvasHeight);
  } else if (render === "3D") {
    summary.innerHTML = "The 3D version visualizes the XOR neural network's predictions as a red-to-yellow graph, where the height and color intensity represent the output. Red indicates values closer to 0, and yellow represents values closer to 1 (or vice-versa), dynamically updating as the network learns in real time."
    canvas = createCanvas(canvasWidth, canvasHeight, WEBGL);
  }
  canvas.parent("canvasContainer"); // Attach canvas to #canvasContainer

  canvas.style("background", "30");
  canvas.style("border-radius", "20px");

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
  background(65, 31, 98);
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
    canvas = null;
  }
  loop();
}

function resetEverything() {
  // Clear previous canvas and points
  clearCanvasVariables();

  // Optionally reset form values or leave them as is
  document.getElementById("canvasForm").reset(); // This will reset the form inputs if needed

  // Disable the canvas
  isCanvasActive = false;

  // Optionally hide the canvas container again
  document.getElementById("canvasContainer").style.display = "block";
}

document
  .getElementById("resetButton")
  .addEventListener("click", function (event) {
    resetEverything();
  });
