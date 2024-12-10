import numpy as np
import cvxpy as cp
from consts import Parameters
from utils import create_boolean_vector, create_block_diag_matrix


class GBDSolver:
    def __init__(self, parameters):
        self.H = parameters['channel']  # Channel Matrix (N x K)
        self.K = parameters['numOfUsers']  # User Number
        self.N = parameters['numOfAntennas']  # Port Number
        self.N_sel = parameters['numOfSelectedAntennas']  # Selected Port Number
        self.gamma = parameters['qosThreshold']  # QoS Threshold
        self.sigmaC2 = parameters['channelNoise']  # Channel Noise
        self.theta = parameters['doa']  # DOA
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
            lambda_old = create_boolean_vector(self.N, self.N_sel)

        # Define CVXPY Variables
        W = cp.Variable((self.N, self.N), complex=True)

        # Constraints
        constraints = []

        # Qos Constraint
        for k in range(self.K):
            lhs = np.sqrt(1 + 1/self.gamma) * cp.real(self.H[:, k:k+1].T @ np.diag(lambda_old[:, 0]) @ W[:, k:k+1])
            rhs = cp.norm(np.vstack(
                [create_block_diag_matrix((self.H[:, k:k+1].T @ np.diag(lambda_old[:, 0])).T, repeat=self.N),
                 np.zeros([1, self.N * self.N])]) @
                          W.reshape((W.size, 1), 'F') +
                          np.vstack([np.zeros([self.N, 1]), np.array([[np.sqrt(self.sigmaC2)]])]), 2)
            constraints.append(rhs - lhs <= 0)

        for k in range(self.K):
            constraints.append(cp.imag(self.H[:, k:k+1].T @ np.diag(lambda_old[:, 0]) @ W[:, k:k+1]) == 0)

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

    def feasibility_check_problem_solver(self, lambda_old=None, verbose=False) -> bool:
        # Initialize Boolean Variable
        if lambda_old is None:
            lambda_old = create_boolean_vector(self.N, self.N_sel)

        # Define CVXPY Variables
        W = cp.Variable((self.N, self.N), complex=True)
        alpha = cp.Variable((self.K, 1))

        # Constraints
        constraints = []

        # Qos Constraint
        for k in range(self.K):
            lhs = np.sqrt(1 + 1/self.gamma) * cp.real(self.H[:, k:k+1].T @ np.diag(lambda_old[:, 0]) @ W[:, k:k+1])
            rhs = cp.norm(np.vstack(
                [create_block_diag_matrix((self.H[:, k:k + 1].T @ np.diag(lambda_old[:, 0])).T, repeat=self.N),
                 np.zeros([1, self.N * self.N])]) @
                          W.reshape((W.size, 1), 'F') +
                          np.vstack([np.zeros([self.N, 1]), np.array([[np.sqrt(self.sigmaC2)]])]), 2)
            constraints.append(rhs - lhs <= alpha[k])

        for k in range(self.K):
            constraints.append(cp.imag(self.H[:, k:k+1].T @ np.diag(lambda_old[:, 0]) @ W[:, k:k+1]) == 0)

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
            obj = cp.real(cp.sum_squares(cp.diag(lambda_old[:, 0]) @ W))
            lagrange_terms = []
            for k in range(self.K):
                lhs = np.sqrt(1 + 1/self.gamma) * cp.real(self.H[:, k:k+1].T @ cp.diag(lambda_old[:, 0]) @ W[:, k:k+1])
                rhs = cp.norm(lambda_old.T @ np.hstack([np.diag(self.H[:, k]) @ W, np.zeros([self.N, 1])]) +
                              np.hstack([np.zeros([1, self.N]), np.array([[np.sqrt(self.sigmaC2)]])]), 2)
                lagrange_terms.append(miu[k, 0] * (rhs - lhs))
            constraints.append(eta >= obj + sum(lagrange_terms))

        if len(self.infeasibilityCut['W']) != 0:
            for j in range(len(self.infeasibilityCut['W'])):
                W = self.infeasibilityCut['W'][j]
                alpha = self.infeasibilityCut['alpha'][j]
                zeta = self.infeasibilityCut['zeta'][j]
                obj = 0
                lagrange_terms = []
                for k in range(self.K):
                    lhs = np.sqrt(1 + 1 / self.gamma) * cp.real(
                        self.H[:, k:k + 1].T @ cp.diag(lambda_old[:, 0]) @ W[:, k:k + 1])
                    rhs = cp.norm(lambda_old.T @ np.hstack([np.diag(self.H[:, k]) @ W, np.zeros([self.N, 1])]) +
                                  np.hstack([np.zeros([1, self.N]), np.array([[np.sqrt(self.sigmaC2)]])]), 2)
                    lagrange_terms.append(zeta[k, 0] * (rhs - lhs - alpha[k, 0]))
                constraints.append(0 >= obj + sum(lagrange_terms))

        # Define the objective function
        objective = cp.Minimize(eta)

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)

        if problem.status == cp.OPTIMAL:
            return lambda_old.value, eta.value
        else:
            raise Exception('Master Problem is Infeasible!')

