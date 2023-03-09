import numpy as np
import scipy

class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """
    
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = np.squeeze(b)

    def func(self, x):
        x = np.squeeze(x)
        return 0.5 * np.dot(np.dot(x, self.A), np.transpose(x)) - np.dot(self.b, np.transpose(x))

    def grad(self, x):
        x = np.squeeze(x)
        return np.squeeze(np.dot(self.A, np.transpose(x)) - np.transpose(self.b))

        
class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i x)) + regcoef * ||x||_2^2.
    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()
    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, A, b, regcoef, matvec_Ax, matvec_ATx, matmat_ATsA):
        self.A = A
        self.b = b
        self.regcoef = regcoef
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA

    def func(self, x):
        x = np.squeeze(x)
        return np.average(np.log(1 + np.exp(-np.multiply(self.b, np.transpose(self.matvec_Ax(x)))))) + np.linalg.norm(x) * self.regcoef

    def grad(self, x):
        x = np.squeeze(x)
        grad = np.zeros_like(x)
        for i in range(np.max(x.shape)):
            grad[i] = np.average(np.multiply(np.multiply(self.b, np.transpose(self.A[:, i])), 1 - 1 / (1 + np.exp(-np.multiply(self.b, np.transpose(self.matvec_Ax(x)))))))
        grad = -grad + self.regcoef * 2 * x

        return grad


def create_log_reg_oracle(A, b, regcoef):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: np.dot(A, np.transpose(x))
    matvec_ATx = lambda x: np.dot(np.transpose(A), x)

    def matmat_ATsA(s):
        return np.dot(np.dot(np.transpose(A), np.diag(s)), A)

    b = np.squeeze(b)

    return LogRegL2Oracle(A, b, regcoef, matvec_Ax, matvec_ATx, matmat_ATsA)
