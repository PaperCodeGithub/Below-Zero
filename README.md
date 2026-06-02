# Below Zero
Exploring Deep Learning from the absolute ground up in pure C.

## About The Repository
Below Zero is a project series where I strip away high level Python abstractions and build Machine Learning architectures from scratch. 
The goal is to understand the raw calculus, memory management, and matrix operations that make artificial intelligence possible. No external libraries.
No automatic differentiation. Just pure systems level programming.

## Projects Included
- **Multi Layer Perceptron:** A complete deep neural network built to classify handwritten digits from the MNIST dataset. Includes custom binary file parsing, forward propagation, loss calculation, and backward pass gradient descent.
- **Word2Vec Skip Gram:** A natural language processing engine that converts raw strings of text into a dense geometric vector space. Demonstrates the sliding context window, softmax probability distribution, and expected versus observed calculus to optimize negative log likelihood.

## Requirements
To run these models, you will need a standard C compiler such as GCC or Clang.

## How to Run
Clone this repository to your local machine. Navigate into either project folder and compile the main C source file.
  
