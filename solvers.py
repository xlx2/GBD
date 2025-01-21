import numpy as np
import cvxpy as cp
from consts import Parameters
from utils import create_boolean_vector, create_block_diag_matrix


class GBDSolver:
    def __init__(self, parameters):
        self.H = parameters['channel']  # Channel Matrix (N x K)
        self.K = parameters['num_of_users']  # User Number
        self.N = parameters['num_of_antennas']  # Port Number
        self.N_sel = parameters['num_of_selected_antennas']  # Selected Antenna Number
        self.L = parameters['snapshot']  # Snapshot Length

        self.gamma = parameters['qos_threshold']  # QoS Threshold
        self.snr_eve = parameters['eve_sensing_threshold']  # Eve Sensing Threshold
        self.tau = parameters['crb_threshold']  # User CRB Threshold
        self.sigmaC2 = parameters['channel_noise']  # Channel Noise
        self.sigmaR2 = parameters['sensing_noise']  # Sensing Noise
        self.beta = parameters['reflection_coefficient'] # Reflection Coefficient

        self.theta = parameters['doa']  # DOA (1 x M)
        a = parameters['steering_vector'] # Steering Vector (N x M)
        a_diff = parameters['diff_steering_vector'] # Steering Vector Diff (N x M)
        self.a_eve = a[:, 0:1]
        self.a_target = a[:, 1:2]
        self.a_eve_diff = a_diff[:, 0:1]
        self.a_target_diff = a_diff[:, 1:2]
        self.A = self.beta * self.a_target @ self.a_target.T.conj()  # Target Steering Matrix (N x N)
        self.A_diff = (self.beta * self.a_target @ self.a_target_diff.T.conj() +
                        self.beta * self.a_target_diff @ self.a_target.T.conj())

        self.feasibilityCut = {
            'W': [],
            'miu': []
        }
        self.infeasibilityCut = {
            'W': [],
            'zeta': [],
            'alpha': []
        }

    def prime_problem_solver(self, lambda_old=None, verbose=False):
        # Initialize Boolean Variable
        if lambda_old is None:
            lambda_old = create_boolean_vector(self.N, self.N_sel)

        # Define CVXPY Variables
        W = cp.Variable((self.N, self.K + self.N), complex=True)
        Q = cp.Variable((self.N, self.K + self.N), complex=True)
        Rx = cp.Variable((self.N, self.N), hermitian=True)

        # Constraints
        constraints = []

        # Qos Constraint
        for k in range(self.K):
            lhs = np.sqrt(1 + 1 / self.gamma) * cp.real(self.H[:, k:k + 1].T @ np.diag(lambda_old[:, 0]) @ W[:, k:k + 1])
            rhs = cp.norm(np.vstack(
                [create_block_diag_matrix((self.H[:, k:k + 1].T @ np.diag(lambda_old[:, 0])).T, repeat=self.K + self.N),
                 np.zeros([1, self.N * (self.K + self.N)])]) @
                          W.reshape((W.size, 1), 'F') +
                          np.vstack([np.zeros([self.K + self.N, 1]), np.array([[np.sqrt(self.sigmaC2)]])]), 2)
            constraints.append(rhs - lhs <= 0)

        # Q = Lambda * W
        constraints.append(Q == np.diag(lambda_old[:, 0]) @ W)

        # Eve Sensing SNR Constraint
        constraints.append(np.abs(self.beta)**2 * cp.sum_squares(self.a_eve.T.conj() @ np.diag(lambda_old[:, 0]) @ W)
                           - self.snr_eve * self.sigmaR2 <= 0)

        # User CRB Constraint
        constraints.append(cp.bmat([
        [cp.trace(self.A_diff.conj().T @ self.A_diff @ Rx) - self.sigmaR2 / (2 * self.L * np.abs(self.beta)**2 * self.tau) , cp.trace(self.A_diff.conj().T @ self.A @ Rx)],
        [cp.trace(self.A_diff.conj().T @ self.A @ Rx), cp.trace(self.A.conj().T @ self.A @ Rx)]]) >> 0)
        constraints.append(Rx >> 0)

        # Imaging Part Constraint
        for k in range(self.K):
            constraints.append(cp.imag(self.H[:, k:k + 1].T @ np.diag(lambda_old[:, 0]) @ W[:, k:k + 1]) == 0)

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
        W = cp.Variable((self.N, self.K + self.N), complex=True)
        Wc = W[:, 0:self.K]
        Wr = W[:, self.K:self.K + self.N]
        alpha = cp.Variable((self.K, 1))

        # Constraints
        constraints = []

        # Qos Constraint
        for k in range(self.K):
            lhs = np.sqrt(1 + 1 / self.gamma) * cp.real(
                self.H[:, k:k + 1].T @ np.diag(lambda_old[:, 0]) @ Wc[:, k:k + 1])
            rhs = cp.norm(np.vstack(
                [create_block_diag_matrix((self.H[:, k:k + 1].T @ np.diag(lambda_old[:, 0])).T, repeat=self.K),
                 np.zeros([1, self.N * self.K])]) @
                          Wc.reshape((Wc.size, 1), 'F') +
                          np.vstack([np.zeros([self.K, 1]), np.array([[np.sqrt(self.sigmaC2)]])]), 2)
            constraints.append(rhs - lhs <= alpha[k])

        for k in range(self.K):
            constraints.append(cp.imag(self.H[:, k:k + 1].T @ np.diag(lambda_old[:, 0]) @ Wc[:, k:k + 1]) == 0)

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

    def master_problem_solver(self, verbose=False):
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
                lhs = np.sqrt(1 + 1 / self.gamma) * cp.real(
                    self.H[:, k:k + 1].T @ cp.diag(lambda_old[:, 0]) @ W[:, k:k + 1])
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


if __name__ == '__main__':
    solver = GBDSolver(Parameters)
    print(solver.prime_problem_solver(verbose=True))
