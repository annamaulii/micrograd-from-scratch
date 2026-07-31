# Micrograd from Scratch

A scalar-valued automatic differentiation engine and multilayer perceptron implemented in Python.

This project was built to understand how neural networks work below high-level frameworks such as PyTorch. It implements computation graphs, reverse-mode automatic differentiation, backpropagation, and gradient descent from first principles.

## Features

- Scalar `Value` objects
- Dynamically constructed computation graphs
- Reverse-mode automatic differentiation
- Automatic backward propagation
- Arithmetic and nonlinear operations
- `Neuron`, `Layer`, and `MLP` components
- Numerical gradient checks
- Unit tests
- XOR training example

## XOR Example

The included MLP learns the nonlinear XOR function:

| Input | Target | Prediction |
|---|---:|---:|
| `[0, 0]` | -1 | -0.958 |
| `[0, 1]` | +1 | 0.940 |
| `[1, 0]` | +1 | 0.944 |
| `[1, 1]` | -1 | -0.954 |

The loss decreased from approximately `7.51` to `0.01` after 300 gradient-descent steps.

## How `backward()` Works

The loss measures how wrong the model's predictions are. Calling `loss.backward()` calculates how sensitive that loss is to every earlier value in the computation graph.

1. Backpropagation starts with `loss.grad = 1.0` because the derivative of the loss with respect to itself is always 1. This provides the starting gradient for propagating backward through earlier nodes.

2. The engine builds a topological ordering so that every node that depends on another node is processed first. Reversing this order ensures gradients propagate backward without processing a node before all its gradient contributions are available.

3. Each node's `_backward()` function applies the chain rule for the operation that created that node. It multiplies the incoming gradient by the operation's local derivative and passes the result to the parent nodes.

4. Gradients use `+=` because one value can influence the loss through multiple paths. Its final gradient must contain the sum of every contribution rather than overwriting earlier contributions.

## Project Structure

```text
micrograd/
├── engine.py    # Scalar values, computation graphs, and backpropagation
└── nn.py        # Neuron, Layer, and MLP components

tests/           # Engine and neural-network tests
examples/        # Runnable training examples
```

- `micrograd/engine.py` implements scalar values, computation graphs, and backpropagation.
- `micrograd/nn.py` implements `Neuron`, `Layer`, and `MLP`.
- `tests/` contains automated engine and neural-network tests.
- `examples/` contains runnable training demonstrations.

## Running the Tests

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

## Running the XOR Example

```bash
python -m examples.train_xor
```

The example trains a small MLP to learn the nonlinear XOR function. It prints the loss during training and the final predictions.

## Acknowledgements

This project was inspired by Andrej Karpathy's
[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

I followed the lecture to learn the core concepts, then organized the implementation as a Python package and added automated tests, numerical gradient checks, and an XOR training example.
