import numpy as np
import cvxpy as cp
from utils import get_boolean_vector, create_block_diag_matrix


class GBDSolver:
    def __init__(self, parameters):
        self.H = parameters['channel']  # Channel Matrix (N x K)
        self.K = parameters['numOfUsers']  # User Number (constant)
        self.N = parameters['numOfAntennas']  # Port Number (constant)
        self.N_sel = parameters[
            'numOfSelectedAntennas']  # Selected Port Number (constant)
        self.gamma = parameters['qosThreshold']  # QoS Threshold (constant)
        self.sigmaC2 = parameters['channelNoise']  # Channel Noise (constant)
        self.theta = parameters['doa']  # DOA (constant)
        self.b = parameters['steeringVector']  # Steering Vector (N x 1)

        self.feasibilityCut = {
            'W': [],
            'miu': []
        }
        self.infeasibilityCut = {
            'W': [],
            'zeta': [],
            'alpha': []
        }

    def prime_problem_solver(self, lambda_old=None, verbose=False) -> (bool, float):
        # Initialize Boolean Variable
        if lambda_old is None:
            lambda_old = get_boolean_vector(self.N, self.N_sel)

        # Define CVXPY Variables
        W = cp.Variable((self.N, self.N), hermitian=True)

        # Constraints
        constraints = []

        # Qos Constraint
        for k in range(self.K):
            lhs = np.sqrt(1 + 1 / self.gamma) * cp.real(self.H[:, k].T @ np.diag(lambda_old[:, 0]) @ W[:, k])
            rhs = cp.norm(create_block_diag_matrix(
                np.repeat((self.H[:, k].T @ np.diag(lambda_old[:, 0])).reshape((self.N, 1)), self.N, axis=1)) @
                          W.reshape((W.size, 1), 'F') +
                          np.vstack((np.zeros((self.N, 1)), np.array([[np.sqrt(self.sigmaC2)]]))), 2)
            constraints.append(rhs - lhs <= 0)

        # Define the objective function
        objective = cp.Minimize(cp.sum_squares(np.diag(lambda_old[:, 0]) @ W))

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)

        # Save the results
        if problem.status == cp.OPTIMAL:
            feasibility = True
            miu = np.array(
                [constraints[k].dual_value for k in range(self.K)]).reshape(-1, 1)
            self.feasibilityCut['W'].append(W.value)
            self.feasibilityCut['miu'].append(miu)
        else:
            feasibility = False

        return feasibility, problem.value

    def feasibility_check_problem_solver(self, lambda_old=None,
                                         verbose=False) -> bool:
        # Initialize Boolean Variable
        if lambda_old is None:
            lambda_old = get_boolean_vector(self.N, self.N_sel)

        # Define CVXPY Variables
        W = cp.Variable((self.N, self.N), hermitian=True)
        alpha = cp.Variable((self.K, 1))

        # Constraints
        constraints = []

        # Qos Constraint
        for k in range(self.K):
            lhs = np.sqrt(1 + 1 / self.gamma) * cp.real(self.H[:, k].T @ np.diag(lambda_old[:, 0]) @ W[:, k])
            rhs = cp.norm(create_block_diag_matrix_with_extra_zero_row(
                np.repeat((self.H[:, k].T @ np.diag(lambda_old[:, 0])).reshape((self.N, 1)), self.N, axis=1)) @
                          W.reshape((W.size, 1), 'F') +
                          np.vstack((np.zeros((self.N, 1)), np.array([[np.sqrt(self.sigmaC2)]]))), 2)
            constraints.append(rhs - lhs <= alpha[k])

        # Non-negative Constraint
        for k in range(self.K):
            constraints.append(alpha[k] >= 0)

        # Define the objective function
        objective = cp.Minimize(cp.sum(alpha))

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)

        # Save the results
        if problem.status == cp.OPTIMAL:
            feasibility = True
            zeta = np.array(
                [constraints[k].dual_value for k in range(self.K)]).reshape(-1, 1)
            self.infeasibilityCut['W'].append(W.value)
            self.infeasibilityCut['zeta'].append(zeta)
            self.infeasibilityCut['alpha'].append(alpha.value)
        else:
            feasibility = False

        return feasibility

    def master_problem_solver(self, verbose=False) -> (np.ndarray, float):
        # Initialize Boolean Variable
        lambda_old = cp.Variable((self.N, 1), boolean=True)
        eta = cp.Variable()

        # Constraints
        constraints = []

        for j in range(len(self.feasibilityCut['W'])):
            W = self.feasibilityCut['W'][j]
            miu = self.feasibilityCut['miu'][j]
            objective = cp.real(cp.sum_squares(cp.diag(lambda_old[:, 0]) @ W))
            lagrange_terms = []
            block_diag_np = np.zeros((self.N, self.N * self.N))
            for n in range(self.N):
                block_diag_np[n, n * self.N:(n + 1) * self.N] = lambda_old.T
            block_diag_cvxpy = cp.Constant(block_diag_np)
            zero_row = cp.Constant(np.zeros((1, self.N * self.N)))
            result = cp.vstack([block_diag_cvxpy, zero_row])
            for k in range(self.K):
                lhs = np.sqrt(1 + 1 / self.gamma) * cp.real(self.H[:, k].T @ cp.diag(lambda_old[:, 0]) @ W[:, k])
                rhs = cp.norm(result @ (np.diag(self.H[:, k]) @ W).reshape((self.N**2, 1), 'F') +
                              np.vstack((np.zeros((self.N, 1)), np.array([[np.sqrt(self.sigmaC2)]]))), 2)
                lagrange_terms.append(miu[k] * (rhs - lhs))
            constraints.append(eta >= objective + sum(lagrange_terms))

        # if len(self.infeasibilityCut['W']) != 0:
        #     for j in range(len(self.infeasibilityCut['W'])):
        #         W1 = self.infeasibilityCut['W'][j]
        #         zeta = self.infeasibilityCut['zeta'][j]
        #         alpha = self.infeasibilityCut['alpha'][j]
        #         Lagrange1 = 0
        #         for k in range(self.K):
        #
        #         constraints.append(0 >= Lagrange1)

        # Define the objective function
        objective = cp.Minimize(eta)

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)

        if problem.status == cp.OPTIMAL:
            return lambda_old.value, eta.value
        else:
            raise Exception('Master Problem is Infeasible!')

