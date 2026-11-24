"""
Implementation Checklist:
    1. Strong Testing
        1. Unit Tests For Integrals etc.
        2. Performance Tests on an Actual Images.
        3. Memory Requirements.
        4. Speed Requirements
        5. Mathematical Foundations
            1. Information Theoretic Lower Bound Arguments

    2. Major Pitches
        1. Compression of 1D and 2D Grids
        2. Image De-pixelation

Organization:
    1. image folders
    2. tests folder
    3. src folders
        - one for laplace
        - one for variational solve

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