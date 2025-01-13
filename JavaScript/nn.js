import Matrix from "./matrixMath.js";

let val = new Matrix(3, 2);
let val2 = new Matrix(3, 2);
val.randomize();
val2.randomize();
val.add(val2);
// val.multiply(8);
val.print();
