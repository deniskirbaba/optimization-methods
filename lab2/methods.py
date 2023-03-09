import numpy as np
from collections import defaultdict
from scipy.optimize.linesearch import scalar_search_wolfe2
from datetime import datetime


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)  # error miss when Armijo -> Wolfe transmission
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c
        elif self._method == 'Armijo':
            # Setting alpha
            alpha = self.alpha_0
            if previous_alpha is not None:
                alpha = previous_alpha * 2  # adaptive term

            # Calculate phi-values
            phi0 = oracle.func_directional(x_k, d_k, 0)
            derphi0 = oracle.grad_directional(x_k, d_k, 0)

            # Check condition and change alpha
            while oracle.func_directional(x_k, d_k, alpha) > phi0 + self.c1 * alpha * derphi0:
                alpha /= 2
            return alpha

        elif self._method == 'Wolfe':
            # Create functions, that calculate phi and derphi
            # to pass them to the wolfe scipy function
            def phi(x):
                return oracle.func_directional(x_k, d_k, x)

            def derphi(x):
                return oracle.grad_directional(x_k, d_k, x)

            # Execute wolfe function
            alpha, *_ = scalar_search_wolfe2(phi=phi, derphi=derphi, c1=self.c1, c2=self.c2)

            # Check if alpha is found, otherwise find alpha using Armijo method
            if alpha is not None:
                return alpha
            else:
                return LineSearchTool(method='Armijo', c1=self.c1, alpha_0=self.alpha_0).line_search(oracle, x_k, d_k,
                                                                                                     previous_alpha)


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


class GradientDescent(object):
    """
    Gradient descent optimization algorithm.
    
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    """

    def __init__(self, oracle, x_0, tolerance=1e-10, line_search_options=None):
        self.oracle = oracle
        x_0 = np.squeeze(x_0)
        self.x_0 = x_0.copy()
        self.tolerance = tolerance
        self.line_search_tool = get_line_search_tool(line_search_options)
        self.hist = defaultdict(list)

    def run(self, max_iter=100):
        """
        Runs gradient descent for max_iter iterations or until stopping 
        criteria is satisfied, starting from point x_0. Saves function values 
        and time in self.hist
        
        self.hist : dictionary of lists
        Dictionary containing the progress information
        Dictionary has to be organized as follows:
            - self.hist['time'] : list of floats, containing time in seconds passed from the start of the method
            - self.hist['func'] : list of function values f(x_k) on every step of the algorithm
            - self.hist['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - self.hist['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
            - self.hist['x_star']: np.array containing x at last iteration

        """
        # Set the changeable variables
        x_cur = np.copy(self.x_0)
        alpha_cur = None

        # Set the stop-condition
        norm_0 = np.power(np.linalg.norm(self.oracle.grad(self.x_0)), 2)
        stop_condition = self.tolerance * norm_0

        # Save starting time
        start = datetime.now()

        # Start main cycle
        for i in range(max_iter):
            # Calculate current direction (grad_cur) and step (alpha_cur)
            d_cur = -self.oracle.grad(x_cur)
            alpha_cur = self.line_search_tool.line_search(self.oracle, x_cur, d_cur,
                                                          alpha_cur if alpha_cur is not None else None)

            # Calculate for stop-condition checking
            norm_cur = np.linalg.norm(-d_cur)

            # Write the history
            if x_cur.size <= 2:
                self.hist['x'].append(np.copy(x_cur))
            self.hist['time'].append((datetime.now() - start).total_seconds())
            self.hist['func'].append(self.oracle.func(x_cur))
            self.hist['grad_norm'].append(np.copy(norm_cur).item(0))

            # Check stop-condition
            if np.power(norm_cur, 2) <= stop_condition:
                self.hist['x_star'].append(np.copy(x_cur))
                return self.hist
            else:
                x_cur += alpha_cur * d_cur

        # If here, then number of iterations exceeded
        # We need to add last state to the history
        if x_cur.size <= 2:
            self.hist['x'].append(np.copy(x_cur))
        self.hist['time'].append((datetime.now() - start).total_seconds())
        self.hist['func'].append(self.oracle.func(x_cur))
        self.hist['grad_norm'].append(np.copy(norm_cur).item(0))
        self.hist['x_star'].append(np.copy(x_cur))

        return self.hist
