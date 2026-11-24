"""


"""

from variational_solver.variational_solver import create_two_dimensional_function
from variational_solver.verification import verification_of_laplacian
import numpy as np
import unittest

def test_create_two_dimensional_function_with_random_data():
    L_x = 5
    L_y = 5
    M = 300
    N = 300
    h = 1
    grid = np.random.rand(int(L_x // h), int(L_y // h))

    # Save the result, so we can fix the results for future.

    u, lambdas = create_two_dimensional_function(h,
                                                 L_x,
                                                 L_y,
                                                 M,
                                                 N,
                                                 grid)

    func = lambda x, y: 0.0
    # Verification of the two d function.
    values_are_close = verification_of_laplacian(u, L_x, L_y, h, lambdas, func, grid, M, N)
    return values_are_close




class TestVariationalFramework(unittest.TestCase):

    def test_variational_framework_random_data_full_quadrature(self):
        calc_correct = test_create_two_dimensional_function_with_random_data()
        self.assertTrue(calc_correct)


if __name__ == '__main__':
    unittest.main()