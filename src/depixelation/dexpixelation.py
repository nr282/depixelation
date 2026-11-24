"""
Code runs the de-pixelation pipeline on the relevant images.

Formulation:
------------

0. Setup:  Add setup of the original problem.
1. Step 1: Formulate the Laplace Problem: grid + boundary.
2. Step 2: Pass the Problem (grid + boundary) to Laplace Equation Solver
3. Step 3: Solve the Laplace Equation, returns a callable.
4. Step 4: Pass the solution, a callable to the problem modifier.
5. Step 5: Modify the problem
6. Step 6: Pass modified problem to Calculus of Variations Solver (solver uses Dirichlet Boundary Conditions)
7. Step 7: Pass the problem to the Poisson Solver with zero boundary.
8. Step 8: Solve the Poisson Equation with zero boundary.
9. Step 9: Solution gets verified for solution of the integral constraints.

"""


from pathlib import Path
import cv2
import os
import numpy as np
from scipy.integrate import dblquad
from variational_solver.variational_solver import create_two_dimensional_function
from laplace_solver.laplace_solver import (calculate_modified_grid,
                                        calculate_modified_grid,
                                        calculate_double_integral,
                                        calculate_laplace_and_grid)
from variational_solver.verification import verification_of_laplacian
import logging


def divide_set(num_dim, i, total_set):


    if i >= num_dim:
        raise ValueError(f"i_x {i} is out of bounds for {num_dim}")

    if num_dim >= total_set:
        raise ValueError(f"num_dim {num_dim} is out of bounds for {total_set}")

    mod = total_set % num_dim
    divisor = total_set // num_dim
    if i < mod:
        low = (divisor + 1) * i
        high = (divisor + 1) * (i + 1)
        return low, high
    else:
        low = (divisor + 1) * mod + divisor * (i - mod)
        high = (divisor + 1) * mod + divisor * ((i + 1) - mod)
        return low, high



class Image(object):

    def __init__(self, image_name: str,
                 image_path: Path = None,
                 image= None):
        self.image_name = image_name
        self.image_path = image_path
        self.cv2_image = image
        if self.cv2_image is None:
            self._load_image()
        self.shape = self.cv2_image.shape
        if len(self.shape) == 3:
            self.x, self.y, self.channels = self.shape
        elif len(self.shape) == 2:
            self.x, self.y = self.shape
            self.channels = 1
        else:
            raise ValueError(f"Image shape is not supported. Shape is {self.shape}")
        self.size = self.cv2_image.size

    def _load_image(self):
        actual_path = os.path.join(os.getcwd(), "image", self.image_path)
        if os.path.exists(actual_path):
            self.cv2_image = cv2.imread(actual_path)
        else:
            raise FileNotFoundError(f"Image {actual_path} does not exist")

    def get_cv2_image(self):
        return self.cv2_image

    def get_shape(self):
        return self.x, self.y, self.channels

    def get_sub_image(self,
                      num_x: int,
                      num_y: int,
                      i_x: int,
                      j_y: int):

        start_row, end_row = divide_set(num_x, i_x,self.x)
        start_col, end_col = divide_set(num_y, j_y,self.y)

        if self.cv2_image is not None:
            sub_image = self.cv2_image[start_row:end_row, start_col:end_col]
            return sub_image

    def __repr__(self):
        return (f"\n*********************************\n"
                f"Image Name: {self.image_name}\n"
                f"Number of Rows: {self.x}\n"
                f"Number of Columns: {self.y}\n"
                f"Number of Pixels: {self.x*self.y}\n"
                f"File Size: {self.size}")


def depixelation(img: Image):
    name = "pixelated_google_meet_image_of_colin"
    img = Image(name,
                "pixelated_image.png")
    engine = DepixelationEngine("depixelate", 10)
    sub_image = img.get_sub_image(100, 100, 3, 3)
    sub_image = Image("pixelated_sub_image", image=sub_image)
    new_img = engine.depixelate(sub_image)

def depixelation_with_boundary_correction():
    """
    Major set of experiments required to develop image
    decompression algorithm.
    """

    name = "pixelated_google_meet_image_of_colin"

    #size of this image is very large.
    img = Image(name,
                "pixelated_image.png")

    M, N = 20, 20
    sub_image = img.cv2_image[:M, :N]
    img = Image("pixelated_sub_image", image=sub_image)
    engine = DepixelationEngine("depixelate", 10)
    new_img_0 = engine.depixelate(img, verification=True)


def calculate_new_grid(new_image,
                       func,
                       u_sol,
                       x,
                       y,
                       ch,
                       h_x,
                       h_y):

    logging.info(f"Calculation of the new grid")

    for i, val_x in enumerate(x):
        for j, val_y in enumerate(y):
            func_quad = calculate_double_integral(func, val_x, val_x + h_x, val_y, val_y + h_y, method="quick")
            u_sol_quad = calculate_double_integral(u_sol, val_x, val_x + h_x, val_y, val_y + h_y, method="quick")
            new_image[i, j, ch] = func_quad + u_sol_quad

    logging.info(f"Finish calculation of the new grid")
    return new_image


class DepixelationEngine(object):

    def __init__(self, algo_name, grid_factor: int):
        self.algo_name = algo_name

    def depixelate(self,
                   image: Image,
                   verification=False):
        """
        Depixelates the image.

        """

        # Frequencies for the transform.
        M = 50
        N = 50

        #Setup spacing
        h=1

        #Setup new grid
        gf = 5
        x = np.linspace(0, image.x, gf*image.x)
        y = np.linspace(0, image.y, gf*image.y)
        h_x = image.x / gf
        h_y = image.y / gf

        #Num Channels
        num = image.channels

        #Setup memory for new image.
        new_image = np.zeros((x.size, y.size, num), dtype=np.float32)


        for ch in range(num):
            grid = image.get_cv2_image()[:, :, ch]
            func, mod_grid = calculate_laplace_and_grid(grid, h, M, N)
            L_x = image.x * h
            L_y = image.y * h

            u_sol, lambdas = create_two_dimensional_function(h,
                                                             L_x,
                                                             L_y,
                                                             M,
                                                             N,
                                                             mod_grid)

            if verification:
                values_are_close = verification_of_laplacian(u_sol, L_x, L_y, h, lambdas, func, grid, M, N)

            calculate_new_grid(new_image,
                               func,
                               u_sol,
                               x,
                               y,
                               ch,
                               h_x,
                               h_y)


        return new_image




if __name__ == "__main__":
    pass