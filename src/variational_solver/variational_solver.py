from typing import Callable, List, Any
import numpy as np
import math
from numpy import ndarray, dtype, float64
from .verification import manual_d2f_dy2, manual_d2f_dx2
from functools import cache
import logging


class MatrixManager(object):
    """
    Matrix Manager to handle fast loading of already developed matrix.

    """

    def __init__(self, m, n):
        pass

@cache
def integral_sin(z, h, m, L):
    return L * (math.cos((np.pi * h * m * z)/L) - math.cos((np.pi * h * m * (z+1)) /L)) / (np.pi * m)

def calculate_u_x_y(x,
                    y,
                    lambdas: np.ndarray,
                    m: int,
                    n: int,
                    L_x: float,
                    L_y: float,
                    h: float,
                    M: int,
                    N: int,
                    coefficients):


    x_curr = x
    y_curr = y

    x = y_curr
    y = x_curr

    res = 0.0
    for m_val in range(1, m+1):
        for n_val in range(1, n+1):
            c_m_n = coefficients[(m_val,n_val)]
            res += c_m_n * math.sin((m_val * x * np.pi) / L_x) * math.sin((n_val * y * np.pi) / L_y)
    return res


def create_two_dimensional_function(h: float,
                                    L_x: float,
                                    L_y: float,
                                    M: int,
                                    N: int,
                                    grid: np.ndarray[tuple[int, int], np.dtype[np.float64]]) -> tuple[
    Callable[[Any, Any], float], ndarray[tuple[Any, ...], dtype[float64]]]:
    """
    Create two-dimensional function based off the input parameters.

    The input parameters correspond to a particular problem.

    The parameters correspond to the problem of:
        1. h - size of the pixel
        2. m - the number of sines in x-axis
        3. n - the number of cosines in y-axis
        4. grid - m x n set of values representing pixel intensity

    Preconditions:
        1. h > 0
        2. m > 0
        3. n > 0
        4. grid[i][j] > 0 for 0 <= i < m, 0 <= j < n

    The grid takes the form:

            [............n..............]
       m    [............n..............]
            [............n..............]

    TODO: Lots of this can be precomputed.
    TODO: Look to refactor this function.

    """

    logging.info("Calculating the two dimensional integral constrained function.")

    grid_m, grid_n = grid.shape
    assert(M > 0 and N > 0)
    assert(L_x / h == grid_m)
    assert (L_y / h == grid_n)

    size = grid_m * grid_n
    A = np.zeros((size,size))
    B = np.zeros(size)

    for r in range(size):
        logging.info(f"Row is provided by: {r}")
        x, y = r // grid_m, r % grid_m
        assert(type(x) == int)
        assert(type(y) == int)
        B[r] = grid[x,y]
        for m in range(1, M + 1):
            for n in range(1, N + 1):
                for x_lam in range(grid_m):
                    for y_lam in range(grid_n):
                        coefficient = 4.0/(L_x * L_y) * 1.0/((m * math.pi / L_x)**2 + (n * math.pi / L_y) **2)
                        index = x_lam * grid_m + y_lam
                        A[r, index] += coefficient * integral_sin(x_lam,h,m,L_x) * integral_sin(y_lam,h,n,L_y) * integral_sin(x,h,m,L_x) * integral_sin(y,h,n,L_y)

    logging.info("Solving the linear system.")
    lambdas = np.linalg.solve(A, B)

    res = 0.0
    coefficients = dict()
    for m_val in range(1, M+1):
        for n_val in range(1, N+1):
            c_m_n = 0
            for x_val in range(grid_m):
                for y_val in range(grid_n):
                    index = x_val * grid_m + y_val
                    lambda_x_y = lambdas[index]
                    c_m_n += lambda_x_y * integral_sin(x_val, h, m_val, L_x) * integral_sin(y_val, h, n_val, L_y)
            c_m_n = c_m_n * 4.0 / (L_x * L_y) * 1.0 / ((m_val * np.pi / L_x) ** 2 + (n_val * np.pi / L_y) ** 2)
            coefficients[(m_val, n_val)] = c_m_n



    fun = lambda x,y: calculate_u_x_y(x,y,lambdas,M,N,L_x,L_y,h, grid_m, grid_n, coefficients)
    return fun, lambdas






