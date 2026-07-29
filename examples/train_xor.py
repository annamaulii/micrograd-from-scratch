import random

from micrograd.nn import MLP


random.seed(42)

inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

targets = [-1.0, 1.0, 1.0, -1.0]

model = MLP(2, [4, 4, 1])

for step in range(300):
    predictions = [model(x) for x in inputs]

    loss = sum(
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
    )

    model.zero_grad()
    loss.backward()

    for parameter in model.parameters():
        parameter.data -= 0.05 * parameter.grad

    if (step + 1) % 50 == 0:
        print(f"Step {step + 1}: loss = {loss.data:.4f}")

print("\nPredictions:")

for x, target in zip(inputs, targets):
    prediction = model(x).data
    print(f"{x} → {prediction:.3f} (target: {target:+.0f})")