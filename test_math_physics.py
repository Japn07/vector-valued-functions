import unittest
import numpy as np

# Import the math engine functions to test
from math_physics import (
    circle_2d, 
    numerical_derivative, 
    tangent_vector, 
    curvature
)

class TestMathPhysics(unittest.TestCase):
    
    def test_circle_2d_evaluation(self):
        """Test basic evaluation of the 2D circle function."""
        # A circle of radius 2 at t=0 should be at (2, 0)
        np.testing.assert_array_almost_equal(circle_2d(0, r=2), [2, 0])
        
        # A circle of radius 2 at t=pi/2 should be at (0, 2)
        np.testing.assert_array_almost_equal(circle_2d(np.pi/2, r=2), [0, 2])

    def test_numerical_derivative(self):
        """Test the numerical derivative against analytical knowns."""
        # The analytical derivative of r(t) = (cos t, sin t) is r'(t) = (-sin t, cos t)
        # At t=0, r'(0) = (0, 1)
        derivative = numerical_derivative(lambda t: circle_2d(t, r=1), 0)
        np.testing.assert_array_almost_equal(derivative, [0, 1], decimal=5)

    def test_tangent_vector_normalization(self):
        """Test that the tangent vector correctly normalizes to length 1."""
        T = tangent_vector(lambda t: circle_2d(t, r=3), 0, normalize=True)
        
        # The unit tangent vector should have a magnitude of 1
        self.assertAlmostEqual(np.linalg.norm(T), 1.0, places=5)
        # At t=0 for a circle, the tangent points straight up along the y-axis
        np.testing.assert_array_almost_equal(T, [0, 1], decimal=5)

    def test_curvature_of_circle(self):
        """Test curvature calculation. A circle of radius r has constant curvature 1/r."""
        radius = 2.0
        c = curvature(lambda t: circle_2d(t, r=radius), 0)
        self.assertAlmostEqual(c, 1.0 / radius, places=4)

if __name__ == '__main__':
    unittest.main()
