import unittest
from functools import partial
from laplace_solver.laplace_solver import (OriginalProblem,
                            LaplaceProblem,
                            LaplaceEquationSolver,
                            Boundary,
                            generate_grid,
                            low_h,
                            high_h,
                            left_v,
                            right_v)
import numpy as np



def calculate_laplace_solution():
    # Generate data
    grid_size = 100
    grid = generate_grid(grid_size)
    h = 0.1

    boundary = Boundary(low_h=partial(low_h, grid, h),
                        high_h=partial(high_h, grid, h),
                        left_v=partial(left_v, grid, h),
                        right_v=partial(right_v, grid, h))

    original_problem = OriginalProblem(grid, h, boundary=boundary)
    problem_size = original_problem.get_problem_size()

    laplace_problem = LaplaceProblem(original_problem.get_boundary(),
                                     h,
                                     original_problem.X,
                                     original_problem.Y,
                                     grid)
    number_of_frequencies = 200
    solver = LaplaceEquationSolver(number_of_frequencies, h, laplace_problem)
    u_sol = solver.solve(laplace_problem)
    verification_good = solver.verification(u_sol, grid=grid)

    return u_sol, original_problem, verification_good



def test_calculate_laplace_solution():

    u_sol, original_problem, verification_good = calculate_laplace_solution()
    return verification_good


class TestVariationalFramework(unittest.TestCase):

    def test_laplace_solution(self):
        self.assertTrue(test_calculate_laplace_solution())




if __name__ == '__main__':
    unittest.main()