import numpy as np
from solver import GBDSolver
from utils import get_boolean_vector

solver = GBDSolver()
lambda_old = get_boolean_vector(solver.N, solver.N_sel)
index = 0
MAX_ITERATIONS = 3
TOLERANCE = 1e-6
UBD = np.inf
LBD = -np.inf
while index < MAX_ITERATIONS and np.abs((UBD - LBD) / max(1, LBD)) > TOLERANCE:
    index += 1
    feasible, optimal = solver.prime_problem_solver(lambda_old, verbose=False)
    if feasible:
        UBD = min(UBD, optimal)
    else:
        solver.feasibility_check_problem_solver(lambda_old, verbose=False)
    lambda_old, eta = solver.master_problem_solver(verbose=False)
    LBD = eta
    print(lambda_old)
    print(UBD, LBD)
