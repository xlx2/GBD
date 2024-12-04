import numpy as np
import cvxpy as cp

from consts import Parameters
from utils import eigenvalue_decomposition, get_boolean_vector


class GBDSolver:
    def __init__(self):
        self.H = Parameters['channel']  # Channel Matrix (N x K)
        self.K = Parameters['numOfUsers']  # User Number (constant)
        self.N = Parameters['numOfAntennas']  # Port Number (constant)
        self.N_sel = Parameters[
            'numOfSelectedAntennas']  # Selected Port Number (constant)
        self.gamma = Parameters['qosThreshold']  # QoS Threshold (constant)
        self.sigmaC2 = Parameters['channelNoise']  # Channel Noise (constant)
        self.theta = Parameters['doa']  # DOA (constant)
        self.b = Parameters['steeringVector']  # Steering Vector (N x 1)

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
        R = [cp.Variable((self.N, self.N), hermitian=True) for _ in
             range(self.K)]
        WWt = cp.Variable((self.N, self.N), hermitian=True)

        # Constraints
        constraints = []

        # Qos Constraint
        for k in range(self.K):
            lhs = cp.real((1 + 1 / self.gamma) * self.H[:, k].T @
                          np.diag(lambda_old[:, 0]) @ R[k] @
                          np.diag(lambda_old[:, 0]) @ np.conj(self.H[:, k]))
            rhs = cp.real(self.H[:, k].T @ np.diag(lambda_old[:, 0]) @ WWt @
                          np.diag(lambda_old[:, 0]) @ np.conj(self.H[:, k])
                          + self.sigmaC2)
            constraints.append(rhs - lhs <= 0)

        # SDR Constraint
        constraints.append(WWt >> 0)
        for k in range(self.K):
            constraints.append(R[k] >> 0)
        constraints.append(WWt - cp.sum(R) >> 0)

        # Define the objective function
        objective = cp.Minimize(cp.real(cp.trace(
            np.diag(lambda_old[:, 0]) @ WWt @ np.diag(lambda_old[:, 0]))))

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)

        if problem.status == cp.OPTIMAL:
            feasible = True
            # Save the results to feasibility cut
            miu = np.array(
                [constraints[k].dual_value for k in range(self.K)]).reshape(-1, 1)
            W = eigenvalue_decomposition(WWt.value)
            self.feasibilityCut['W'].append(W)
            self.feasibilityCut['miu'].append(miu)
        else:
            feasible = False

        return feasible, problem.value

    def feasibility_check_problem_solver(self, lambda_old=None,
                                         verbose=False) -> bool:
        # Initialize Boolean Variable
        if lambda_old is None:
            lambda_old = get_boolean_vector(self.N, self.N_sel)

        # Define CVXPY Variables
        R = [cp.Variable((self.N, self.N), hermitian=True) for _ in
             range(self.K)]
        WWt = cp.Variable((self.N, self.N), hermitian=True)
        alpha = cp.Variable((self.K, 1))

        # Constraints
        constraints = []

        # Qos Constraint
        for k in range(self.K):
            lhs = cp.real((1 + 1 / self.gamma) * self.H[:, k].T @
                          np.diag(lambda_old[:, 0]) @ R[k] @
                          np.diag(lambda_old[:, 0]) @ np.conj(self.H[:, k]))
            rhs = cp.real(self.H[:, k].T @ np.diag(lambda_old[:, 0]) @ WWt @
                          np.diag(lambda_old[:, 0]) @ np.conj(self.H[:, k])
                          + self.sigmaC2)
            constraints.append(rhs - lhs <= alpha[k])

        # Non-negative Constraint
        for k in range(self.K):
            constraints.append(alpha[k] >= 0)

        # SDR Constraint
        constraints.append(WWt >> 0)
        for k in range(self.K):
            constraints.append(R[k] >> 0)
        constraints.append(WWt - cp.sum(R) >> 0)

        # Define the objective function
        objective = cp.Minimize(cp.sum(alpha))

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)

        if problem.status == cp.OPTIMAL:
            feasible = True
            # Save the results to infeasibility cut
            zeta = np.array(
                [constraints[k].dual_value for k in range(self.K)]).reshape(-1, 1)
            W = eigenvalue_decomposition(WWt.value)
            self.infeasibilityCut['W'].append(W)
            self.infeasibilityCut['zeta'].append(zeta)
            self.infeasibilityCut['alpha'].append(alpha.value)
        else:
            feasible = False

        return feasible

    def master_problem_solver(self, verbose=False) -> (np.ndarray, float):
        # Initialize Boolean Variable
        Lambda = cp.Variable((self.N, 1), boolean=True)
        eta = cp.Variable()

        # Constraints
        constraints = []

        for j in range(len(self.feasibilityCut['W'])):
            W = self.feasibilityCut['W'][j]
            miu = self.feasibilityCut['miu'][j]
            Lagrange = cp.real(cp.norm(Lambda[:, 0] @ W, 'fro'))
            for k in range(self.K):
                R = (1 + 1 / self.gamma) * W[:, k] @ np.conj(W[:, k].T) - W @ np.conj(W.T)
                Lagrange = Lagrange + miu[k, 0]**2 * cp.real(np.sqrt(self.sigmaC2) - cp.norm(
                    self.H[:, k].T @ cp.diag(Lambda[:, 0]) @
                    eigenvalue_decomposition(R), 2))
            constraints.append(eta >= Lagrange)

        if len(self.infeasibilityCut['W']) != 0:
            for j in range(len(self.infeasibilityCut['W'])):
                W1 = self.infeasibilityCut['W'][j]
                zeta = self.infeasibilityCut['zeta'][j]
                alpha = self.infeasibilityCut['alpha'][j]
                Lagrange1 = 0
                for k in range(self.K):
                    R = (1 + 1 / self.gamma) * W1[:, k] @ np.conj(W1[:, k].T) - W1 @ np.conj(W1.T)
                    Lagrange1 = Lagrange1 + zeta[k, 0]**2 * (np.sqrt(self.sigmaC2 - alpha[k, 0])
                                - cp.real(cp.norm(self.H[:, k].T @ cp.diag(Lambda[:, 0]) @
                        eigenvalue_decomposition(R), 2)))
                constraints.append(0 >= Lagrange1)

        # Define the objective function
        objective = cp.Minimize(eta)

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=verbose)

        if problem.status == cp.OPTIMAL:
            lambda_old = Lambda.value
            eta = eta.value
            return lambda_old, eta
        else:
            raise Exception('Master Problem is Infeasible!')


if __name__ == '__main__':
    solver = GBDSolver()
    lambda_old = get_boolean_vector(solver.N, solver.N_sel)
    feasible, obj = solver.prime_problem_solver(lambda_old, verbose=True)
    print(feasible, obj)