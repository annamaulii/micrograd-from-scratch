import unittest
import random
from micrograd.engine import Value
from micrograd.nn import Neuron
from micrograd.nn import MLP


class TestNeuron(unittest.TestCase):
    def test_neuron_forward_pass(self):
        neuron = Neuron(3)

        output = neuron([2.0, 3.0, -1.0])

        self.assertIsInstance(output, Value)
        self.assertEqual(len(neuron.parameters()), 4)

class TestMLP(unittest.TestCase):
    def test_mlp_forward_pass_and_parameters(self):
        model = MLP(3, [4, 4, 1])

        output = model([2.0, 3.0, -1.0])

        self.assertIsInstance(output, Value)
        self.assertEqual(len(model.parameters()), 41)

    def test_training_step_reduces_loss(self):
        random.seed(42)
        model = MLP(3, [4, 4, 1])

        x = [2.0, 3.0, -1.0]
        target = 1.0

        prediction = model(x)
        loss = (prediction - target) ** 2
        loss_before = loss.data

        loss.backward()

        for parameter in model.parameters():
            parameter.data -= 0.01 * parameter.grad

        new_prediction = model(x)
        new_loss = (new_prediction - target) ** 2

        self.assertLess(new_loss.data, loss_before)

    def test_zero_grad(self):
        random.seed(42)
        model = MLP(3, [4, 4, 1])

        prediction = model([2.0, 3.0, -1.0])
        loss = (prediction - 1.0) ** 2
        loss.backward()

        self.assertTrue(
            any(parameter.grad != 0.0 for parameter in model.parameters())
        )

        model.zero_grad()

        self.assertTrue(
            all(parameter.grad == 0.0 for parameter in model.parameters())
        )