
from typing import Callable, List
import numpy as np
import math
from collections.abc import Callable
from typing import Protocol
from functools import partial
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
import numpy as np
import logging
from enum import Enum

logging.basicConfig(level=logging.DEBUG)


USE_ANALYTICAL = True


class DIRECTION(Enum):
    vertical = 0
    horizontal = 1


class EDGE(Enum):
    low = 0
    high = 1


class Boundary(object):
    """
    A boundary represents the edges of the grid.

    For instance, if the grid is provided by the domain:
        grid = [0,M] x [0,N]

    There are four boundaries represented below:

        -------------------  (high_h)
        |                  |
        |                  |
        |                  |
        |                  |
        |                  |
        -------------------  (low_h)
    left_v              right_v
    """

    def __init__(self,
                 low_h,
                 high_h,
                 left_v,
                 right_v):

        self.low_h = low_h
        self.high_h = high_h
        self.left_v = left_v
        self.right_v = right_v



def generate_grid(n) -> np.ndarray:

    grid = np.random.rand(n,n)
    return grid


def manual_d2f_dx2(func, x, y, h=1e-4):
    return (func(x + h, y) - 2 * func(x, y) + func(x - h, y)) / (h**2)

def manual_d2f_dy2(func, x, y, h=1e-4):
    return (func(x, y+h) - 2 * func(x, y) + func(x, y-h)) / (h**2)

class LaplaceProblem(object):

    def __init__(self, boundary: Boundary, h: float, X: float, Y: float, grid):
        self.boundary = boundary
        self.h = h
        self.X = X
        self.Y = Y
        self.grid = grid


class BoundaryManager(object):

    def __init__(self, f, direction: DIRECTION, edge: EDGE, grid, h: float):
        m, n = grid.shape
        self.f = f
        self.h = h
        if direction == DIRECTION.horizontal and edge == EDGE.low:
            self.one_d_grid = [grid[0,i] for i in range(n)]
        elif direction == DIRECTION.horizontal and edge == EDGE.high:
            self.one_d_grid = [grid[m-1,i] for i in range(n)]
        elif direction == DIRECTION.vertical and edge == EDGE.low:
            self.one_d_grid = [grid[i,0] for i in range(m)]
        elif direction == DIRECTION.vertical and edge == EDGE.high:
            self.one_d_grid = [grid[i,n-1] for i in range(m)]


    def __call__(self, x):
        return self.f(x)


def low_h(grid, h: float, x: int):

    m,n=grid.shape

    j = int(x//h)
    if j < n:
        return grid[0, j]
    else:
        raise

def high_h(grid, h_step, x):

    m, n = grid.shape
    j = int(x // h_step)
    if j < n:
        return grid[len(grid)-1,j]
    else:
        raise

def left_v(grid, h, y):
    m, n = grid.shape
    i = int(y // h)
    if i < m:
        return grid[i,0]
    else:
        raise

def right_v(grid, h, y):

    m, n = grid.shape
    i = int(y // h)
    if i < m:
        return grid[i, len(grid[0])-1]
    else:
        raise


def sin_integral(a: float, b: float, denom: float, n: int):
    """
    Calculate integral of:

    np.sin((n * np.pi * x) / denom) from a to b.

    is:

        denom / (n * pi) * (cos((n pi a) / denom) - cos((n pi b) / denom))

    """

    return (denom / (n * np.pi)) * (np.cos((n * np.pi * a) / denom) - np.cos((n * np.pi * b) / denom))


def calculate_double_integral(func, x_low, x_high, y_low, y_high, method="quick"):

    if method == "quad":
        area, error = dblquad(func, x_low, x_high, y_low, y_high, epsrel=1e-1, epsabs=1e-1)
        return area
    elif method == "quick":
        return func((x_high + x_low) / 2, (y_high + y_low) / 2)




def calculate_modified_grid(original_problem, u_sol: Callable):

    # Use Laplace Solution to modify the grid.
    m, n = original_problem.grid.shape
    mod_grid = original_problem.grid.copy()
    for i in range(m):
        for j in range(n):
            logging.debug(f"Calculating Integral for Index {i} and index {j}".format(i=i, j=j))
            val, x_bounds, y_bounds = original_problem.get_value(i, j)
            x_low, x_high = x_bounds
            y_low, y_high = y_bounds
            integral = calculate_double_integral(u_sol, x_low, y_low, x_high, y_high)
            mod_grid[i][j] = mod_grid[i][j] - integral

    return mod_grid


class OriginalProblem(object):

    def __init__(self,
                 grid: np.ndarray,
                 h: float,
                 boundary=None,
                 M=50,
                 N=50):
        """
        The grid is provided below:

            [[2,3,2],
            [3,4,5],
            [4,5,6]]


        :param grid: np.array
        :param h: float, size of the grid
        """

        self.grid = grid
        m, n = self.grid.shape
        self.boundary = boundary
        self.h = h
        self.m = m
        self.n = n
        self.X = m * h
        self.Y = n * h


    def get_value(self, i, j):

        if 0 <= i and i < self.m and 0 <= j and j < self.n:
            return self.grid[i][j], (i*self.h, (i+1) * self.h), (j*self.h, (j+1)* self.h)
        else:
            raise ValueError("i and j are not in bounds")

    def get_boundary(self):
        """
        Returns boundary Function
        :return:
        """

        return self.boundary


    def add_mean_boundary(self, mean_boundary):
        self.boundary = mean_boundary


    def get_problem_size(self):

        return f"[0,{self.X}] x [0,{self.Y}]"

class LaplaceEquationSolver(object):

    def __init__(self, N, h: float, problem: LaplaceProblem):
        self.N = N
        self.h = h
        self.problem = problem

    def calculate_fourier_coefficient_for_lower(self, f, n: int, use_analytical=USE_ANALYTICAL):

        def integrand(x):
            return f(x) * np.sin((n * np.pi * x) / self.problem.X)

        alpha = (2 / (self.problem.X * np.sinh((n * np.pi * self.problem.Y / self.problem.X))))
        if not use_analytical:
            numerical_value = alpha * quad(integrand, 0, self.problem.X)[0]
            return numerical_value
        else:
            res = 0
            for i in range(len(f.one_d_grid)):
                res += f.one_d_grid[i] * sin_integral(i * f.h, (i+1) * f.h, self.problem.X, n)
            analytical_value = alpha * res
            return analytical_value

    def _solve_lower_boundary(self, f) -> Callable[[float, float], float]:


        def u_1(x,y):
            res = 0
            for n in range(1, self.N + 1):
                a_n = self.calculate_fourier_coefficient_for_lower(f, n)
                #logging.debug(f"An is provided by {a_n}")
                val = np.sinh(n * np.pi * (self.problem.Y - y) / self.problem.X) * np.sin((n*np.pi*x)/self.problem.X)
                if not math.isfinite(val):
                    raise

                res += a_n * val

            return res

        return u_1



    def calculate_fourier_coefficient_for_upper(self, f: BoundaryManager, n: int, use_analytical=USE_ANALYTICAL):


        def integrand(x):

            return f(x) * np.sin((n * np.pi * x) / self.problem.X)

        alpha = (2 / (self.problem.X * np.sinh((n * np.pi * self.problem.Y / self.problem.X) )))
        if not use_analytical:
            return alpha * quad(integrand, 0, self.problem.X)[0]
        else:
            res = 0
            for i in range(len(f.one_d_grid)):
                res += f.one_d_grid[i] * sin_integral(i * f.h, (i + 1) * f.h, self.problem.X, n)
            analytical_value = alpha * res
            return analytical_value

    def _solve_upper_boundary(self, f) -> Callable[[float, float], float]:


        def u_2(x,y):
            res = 0
            for n in range(1, self.N + 1):
                b_n = self.calculate_fourier_coefficient_for_upper(f, n)
                #logging.debug(f"Bn is provided by {b_n}")
                val = np.sinh((n * np.pi * y) / self.problem.X) * np.sin((n * np.pi * x) / self.problem.X)

                if not math.isfinite(val):
                    raise

                res += b_n * val

            return res

        return u_2


    def calculate_fourier_coefficient_for_left(self, f, n: int, use_analytical=USE_ANALYTICAL):

        def integrand(y):
            return f(y) * np.sin((n * np.pi * y) / self.problem.Y)


        alpha = (2 / (self.problem.Y * np.sinh((n * np.pi * self.problem.X / self.problem.Y))))

        if not use_analytical:
            return 2 / (self.problem.Y * np.sinh((n * np.pi * self.problem.X / self.problem.Y))) * \
                quad(integrand, 0, self.problem.Y)[0]
        else:
            res = 0
            for i in range(len(f.one_d_grid)):
                res += f.one_d_grid[i] * sin_integral(i * f.h, (i + 1) * f.h, self.problem.Y, n)
            analytical_value = alpha * res
            return analytical_value


    def _solve_left_boundary(self, f) -> Callable[[float, float], float]:

        def u_3(x,y):
            res = 0
            for n in range(1, self.N + 1):
                c_n = self.calculate_fourier_coefficient_for_left(f, n)
                val = np.sinh((n * np.pi * (self.problem.X - x)) / self.problem.Y) * np.sin((n*np.pi*y)/self.problem.Y)

                if not math.isfinite(val):
                    raise

                #logging.debug(f"Cn is provided by {c_n}")
                res += c_n * val

            return res

        return u_3

    def calculate_fourier_coefficient_for_right(self, f, n: int, use_analytical=USE_ANALYTICAL):

        def integrand(y):
            return f(y) * np.sin((n * np.pi * y) / self.problem.Y)

        alpha = (2 / (self.problem.Y * np.sinh((n * np.pi * self.problem.X / self.problem.Y))))
        if not use_analytical:
            return alpha * quad(integrand, 0, self.problem.Y)[0]
        else:
            res = 0
            for i in range(len(f.one_d_grid)):
                res += f.one_d_grid[i] * sin_integral(i * f.h, (i + 1) * f.h, self.problem.Y, n)
            analytical_value = alpha * res
            return analytical_value

    def _solve_right_boundary(self, f) -> Callable[[float, float], float]:

        def u_4(x,y):
            res = 0
            for n in range(1, self.N + 1):
                d_n = self.calculate_fourier_coefficient_for_right(f, n)
                val = np.sinh((n*np.pi*x)/self.problem.Y) * np.sin((n*np.pi*y)/self.problem.Y)
                if not math.isfinite(val):
                    raise

                #logging.debug(f"Dn is provided by {d_n}")
                res += d_n * val

            return res

        return u_4

    def _setup_boundary(self, problem):

        upper_boundary = BoundaryManager(problem.boundary.high_h,
                                         DIRECTION.horizontal,
                                         EDGE.high,
                                         problem.grid,
                                         self.h)
        lower_boundary = BoundaryManager(problem.boundary.low_h,
                                         DIRECTION.horizontal,
                                         EDGE.low,
                                         problem.grid,
                                         self.h)

        left_boundary = BoundaryManager(problem.boundary.left_v,
                                        DIRECTION.vertical,
                                        EDGE.low,
                                        problem.grid,
                                        self.h)

        right_boundary = BoundaryManager(problem.boundary.right_v,
                                         DIRECTION.vertical,
                                         EDGE.high,
                                         problem.grid,
                                         self.h)

        return upper_boundary, lower_boundary, left_boundary, right_boundary


    def solve(self,
              problem: LaplaceProblem,
              ) -> Callable[[float, float], float]:
        """
        Solves the laplace problem.

        The Laplace Equation for a 2D rectangular domain:

        u_xx + u_yy = 0


        :param problem:
        :return:
        """

        h = problem.h

        upper_boundary, lower_boundary, left_boundary, right_boundary = self._setup_boundary(problem)


        u_upper = self._solve_upper_boundary(upper_boundary)
        u_lower = self._solve_lower_boundary(lower_boundary)
        u_left = self._solve_left_boundary(left_boundary)
        u_right = self._solve_right_boundary(right_boundary)



        def result(x: float, y: float):
            return u_upper(x,y) + u_lower(x,y) + u_left(x,y) + u_right(x,y)


        return result


    def verification(self,
                     f: Callable[[float, float], float],
                     grid=None,
                     epsilon=1e-3):
        """
        Verification requires:
            1. Laplacian is zero on the domain
            2. Laplacian aligns with the boundary


        :return:
        """

        del_calcs = []
        for i in range(100):
            x,y = np.random.uniform(low=0, high=self.problem.X), np.random.uniform(low=0, high=self.problem.Y)
            del_calc = manual_d2f_dx2(f,x,y) + manual_d2f_dy2(f,x,y)
            del_calcs.append(del_calc)
        interior_is_zero =  all([del_calc < epsilon for del_calc in del_calcs])
        return interior_is_zero


class CalculusOfVariationsProblem(object):

    def __init__(self,
                 modified_grid: np.ndarray,
                 original_problem: OriginalProblem):
        self.h = original_problem.h
        self.X = original_problem.X
        self.Y = original_problem.Y
        self.modified_grid = modified_grid


def calculate_modified_grid(original_problem, u_sol: Callable):

    # Use Laplace Solution to modify the grid.
    m, n = original_problem.grid.shape
    mod_grid = original_problem.grid.copy()
    for i in range(m):
        for j in range(n):
            logging.debug(f"Calculating Integral for Index {i} and index {j}".format(i=i, j=j))
            val, x_bounds, y_bounds = original_problem.get_value(i, j)
            x_low, x_high = x_bounds
            y_low, y_high = y_bounds
            integral = calculate_double_integral(u_sol, x_low, y_low, x_high, y_high, method="quick")
            mod_grid[i][j] = mod_grid[i][j] - integral

    return mod_grid


def calculate_laplace_solution(grid = None,
                               h=None,
                               number_of_frequencies = 200,
                               M=20,
                               N=20):

    if grid is None and h is None:
        # Generate data
        grid_size = 100
        grid = generate_grid(grid_size)
        h = 0.1



    boundary = Boundary(low_h=partial(low_h, grid, h),
                        high_h=partial(high_h, grid, h),
                        left_v=partial(left_v, grid, h),
                        right_v=partial(right_v, grid, h))

    original_problem = OriginalProblem(grid, h, boundary=boundary, M=M, N=N)
    problem_size = original_problem.get_problem_size()

    laplace_problem = LaplaceProblem(original_problem.get_boundary(),
                                     h,
                                     original_problem.X,
                                     original_problem.Y,
                                     grid)

    solver = LaplaceEquationSolver(number_of_frequencies, h, laplace_problem)
    u_sol = solver.solve(laplace_problem)
    solver.verification(u_sol, grid=grid)

    return u_sol, original_problem

def calculate_laplace_and_grid(grid = None, h=None, M=None, N=None):

    u_sol, original_problem = calculate_laplace_solution(grid=grid, h=h, M=M, N=N)
    modified_grid = calculate_modified_grid(original_problem, u_sol)
    return u_sol, modified_grid


