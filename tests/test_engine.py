import unittest
import math

from micrograd.engine import Value


class TestValue(unittest.TestCase):
    def test_stores_data(self):
        value = Value(2.0)

        self.assertEqual(value.data, 2.0)

    def test_forward_expression(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        f = Value(-2.0)

        result = (a * b + c) * f

        self.assertAlmostEqual(result.data, -8.0)

    def test_backward_expression(self):
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        f = Value(-2.0)

        loss = (a * b + c) * f
        loss.backward()

        self.assertAlmostEqual(a.grad, 6.0)
        self.assertAlmostEqual(b.grad, -4.0)
        self.assertAlmostEqual(c.grad, -2.0)
        self.assertAlmostEqual(f.grad, 4.0)
        self.assertAlmostEqual(loss.grad, 1.0)

    def test_reused_value_accumulates_gradient(self):
        a = Value(3.0)

        result = a + a
        result.backward()

        self.assertAlmostEqual(a.grad, 2.0)

    def test_gradient_matches_finite_difference(self):
        h = 0.000001

        a = Value(2.0)
        loss = (a * -3.0 + 10.0) * -2.0
        loss.backward()

        baseline = (2.0 * -3.0 + 10.0) * -2.0
        perturbed = ((2.0 + h) * -3.0 + 10.0) * -2.0
        numerical_gradient = (perturbed - baseline) / h

        self.assertAlmostEqual(a.grad, numerical_gradient, places=5)

    def test_tanh_gradient_matches_finite_difference(self):
        h = 0.000001
        x_value = 0.5

        x = Value(x_value)
        result = x.tanh()
        result.backward()

        baseline = math.tanh(x_value)
        perturbed = math.tanh(x_value + h)
        numerical_gradient = (perturbed - baseline) / h

        self.assertAlmostEqual(x.grad, numerical_gradient, places=5)