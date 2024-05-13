import warnings
from itertools import combinations as combo_nr
from itertools import product
from itertools import repeat
from math import comb
from typing import cast
from typing import NewType

import cvxpy as cp
import numpy as np
from numpy.typing import NBitBase
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning

from ..feature_library.polynomial_library import PolynomialLibrary
from ..utils import reorder_constraints
from .sr3 import SR3

AnyFloat = np.dtype[np.floating[NBitBase]]
Int1D = np.ndarray[tuple[int], np.dtype[np.int_]]
Float2D = np.ndarray[tuple[int, int], AnyFloat]
Float3D = np.ndarray[tuple[int, int, int], AnyFloat]
Float4D = np.ndarray[tuple[int, int, int, int], AnyFloat]
Float5D = np.ndarray[tuple[int, int, int, int, int], AnyFloat]
FloatND = NDArray[np.floating[NBitBase]]
NFeat = NewType("NFeat", int)
NTarget = NewType("NTarget", int)


class TrappingSR3(SR3):
    """
    Generalized trapping variant of sparse relaxed regularized regression.
    This optimizer can be used to identify quadratically nonlinear systems with
    either a-priori globally or locally stable (bounded) solutions.

    This optimizer can be used to minimize five different objective functions:

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(w)
        + 0.5\\|Pw-A\\|^2_2/\\eta + \\delta_0(Cw-d)
        + \\delta_{\\Lambda}(A) + \\alpha \\|Qijk\\|
        + \\beta \\|Q_{ijk} + Q_{jik} + Q_{kij}\\|

    where :math:`R(w)` is a regularization function, C is a constraint matrix
    detailing affine constraints on the model coefficients, A is a proxy for
    the quadratic contributions to the energy evolution, and
    :math:`Q_{ijk}` are the quadratic coefficients in the model. For
    provably globally bounded solutions, use :math:`\\alpha >> 1`,
    :math:`\\beta >> 1`, and equality constraints. For maximizing the local
    stability radius of the model one has the choice to do this by
    (1) minimizing the values in :math:`Q_{ijk}`, (2) promoting models
    with skew-symmetrix :math:`Q_{ijk}` coefficients, or
    (3) using inequality constraints for skew-symmetry in :math:`Q_{ijk}`.

    See the following references for more details:

        Kaptanoglu, Alan A., et al. "Promoting global stability in
        data-driven models of quadratic nonlinear dynamics."
        arXiv preprint arXiv:2105.01843 (2021).

        Zheng, Peng, et al. "A unified framework for sparse relaxed
        regularized regression: Sr3." IEEE Access 7 (2018): 1404-1423.

        Champion, Kathleen, et al. "A unified sparse optimization framework
        to learn parsimonious physics-informed models from data."
        IEEE Access 8 (2020): 169259-169271.

    Parameters
    ----------
    evolve_w : bool, optional (default True)
        If false, don't update w and just minimize over (m, A)

    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the L0 norm, the regularization
        is equivalent to performing hard thresholding, and lambda
        is chosen to threshold at the value given by this parameter.
        This is equivalent to choosing lambda = threshold^2 / (2 * nu).

    eta : float, optional (default 1.0e20)
        Determines the strength of the stability term ||Pw-A||^2 in the
        optimization. The default value is very large so that the
        algorithm default is to ignore the stability term. In this limit,
        this should be approximately equivalent to the ConstrainedSR3 method.

    alpha_m : float, optional (default eta * 0.1)
        Determines the step size in the prox-gradient descent over m.
        For convergence, need alpha_m <= eta / ||w^T * PQ^T * PQ * w||.
        Typically 0.01 * eta <= alpha_m <= 0.1 * eta.

    alpha_A : float, optional (default eta)
        Determines the step size in the prox-gradient descent over A.
        For convergence, need alpha_A <= eta, so typically
        alpha_A = eta is used.

    alpha : float, optional (default 1.0e20)
        Determines the strength of the local stability term ||Qijk||^2 in the
        optimization. The default value is very large so that the
        algorithm default is to ignore this term.

    beta : float, optional (default 1.0e20)
        Determines the strength of the local stability term
        ||Qijk + Qjik + Qkij||^2 in the
        optimization. The default value is very large so that the
        algorithm default is to ignore this term.

    gamma : float, optional (default 0.1)
        Determines the negative interval that matrix A is projected onto.
        For most applications gamma = 0.1 - 1.0 works pretty well.

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm over w.

    tol_m : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm over m.

    thresholder : string, optional (default 'L1')
        Regularization function to use. For current trapping SINDy,
        only the L1 and L2 norms are implemented. Note that other convex norms
        could be straightforwardly implemented, but L0 requires
        reformulation because of nonconvexity.

    thresholds : np.ndarray, shape (n_targets, n_features), optional \
            (default None)
        Array of thresholds for each library function coefficient.
        Each row corresponds to a measurement variable and each column
        to a function from the feature library.
        Recall that SINDy seeks a matrix :math:`\\Xi` such that
        :math:`\\dot{X} \\approx \\Theta(X)\\Xi`.
        ``thresholds[i, j]`` should specify the threshold to be used for the
        (j + 1, i + 1) entry of :math:`\\Xi`. That is to say it should give the
        threshold to be used for the (j + 1)st library function in the equation
        for the (i + 1)st measurement variable.

    eps_solver : float, optional (default 1.0e-7)
        If threshold != 0, this specifies the error tolerance in the
        CVXPY (OSQP) solve. Default is 1.0e-3 in OSQP.

    inequality_constraints : bool, optional (default False)
        If True, CVXPY methods are used.

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    accel : bool, optional (default False)
        Whether or not to use accelerated prox-gradient descent for (m, A).

    m0 : np.ndarray, shape (n_targets), optional (default None)
        Initial guess for vector m in the optimization. Otherwise
        each component of m is randomly initialized in [-1, 1].

    A0 : np.ndarray, shape (n_targets, n_targets), optional (default None)
        Initial guess for vector A in the optimization. Otherwise
        A is initialized as A = diag(gamma).

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every iteration.

    verbose_cvxpy : bool, optional (default False)
        Boolean flag which is passed to CVXPY solve function to indicate if
        output should be verbose or not. Only relevant for optimizers that
        use the CVXPY package in some capabity.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    history_ : list
        History of sparse coefficients. ``history_[k]`` contains the
        sparse coefficients (v in the optimization objective function)
        at iteration k.

    objective_history_ : list
        History of the value of the objective at each step. Note that
        the trapping SINDy problem is nonconvex, meaning that this value
        may increase and decrease as the algorithm works.

    A_history_ : list
        History of the auxiliary variable A that approximates diag(PW).

    m_history_ : list
        History of the shift vector m that determines the origin of the
        trapping region.

    PW_history_ : list
        History of PW = A^S, the quantity we are attempting to make
        negative definite.

    PWeigs_history_ : list
        History of diag(PW), a list of the eigenvalues of A^S at
        each iteration. Tracking this allows us to ascertain if
        A^S is indeed being pulled towards the space of negative
        definite matrices.

    PL_unsym_ : np.ndarray, shape (n_targets, n_targets, n_targets, n_features)
        Unsymmetrized linear coefficient part of the P matrix in ||Pw - A||^2

    PL_ : np.ndarray, shape (n_targets, n_targets, n_targets, n_features)
        Linear coefficient part of the P matrix in ||Pw - A||^2

    PQ_ : np.ndarray, shape (n_targets, n_targets,
                            n_targets, n_targets, n_features)
        Quadratic coefficient part of the P matrix in ||Pw - A||^2

    PT_ : np.ndarray, shape (n_targets, n_targets,
                            n_targets, n_targets, n_features)
        Transpose of 1st dimension and 2nd dimension of quadratic coefficient
        part of the P matrix in ||Pw - A||^2

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import TrappingSR3
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = TrappingSR3(threshold=0.1)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.004 1 + 10.004 x0
    x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
    x2' = -2.662 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        evolve_w=True,
        threshold=0.1,
        eps_solver=1e-7,
        inequality_constraints=False,
        eta=None,
        alpha=None,
        beta=None,
        mod_matrix=None,
        alpha_A=None,
        alpha_m=None,
        gamma=-0.1,
        tol=1e-5,
        tol_m=1e-5,
        thresholder="l1",
        thresholds=None,
        max_iter=30,
        accel=False,
        normalize_columns=False,
        fit_intercept=False,
        copy_X=True,
        m0=None,
        A0=None,
        objective_history=None,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
        verbose=False,
        verbose_cvxpy=False,
    ):
        super(TrappingSR3, self).__init__(
            threshold=threshold,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            thresholder=thresholder,
            thresholds=thresholds,
            verbose=verbose,
        )
        if thresholder.lower() not in ("l1", "l2", "weighted_l1", "weighted_l2"):
            raise ValueError("Regularizer must be (weighted) L1 or L2")
        if eta is None:
            warnings.warn(
                "eta was not set, so defaulting to eta = 1e20 "
                "with alpha_m = 1e-2 * eta, alpha_A = eta. Here eta is so "
                "large that the stability term in the optimization "
                "will be ignored."
            )
            eta = 1e20
            alpha_m = 1e18
            alpha_A = 1e20
        else:
            if alpha_m is None:
                alpha_m = eta * 1e-2
            if alpha_A is None:
                alpha_A = eta
        if eta <= 0:
            raise ValueError("eta must be positive")
        if alpha is None:
            alpha = 1e20
            warnings.warn(
                "alpha was not set, so defaulting to alpha = 1e20 "
                "which is so"
                "large that the ||Qijk|| term in the optimization "
                "will be essentially ignored."
            )
        if beta is None:
            beta = 1e20
            warnings.warn(
                "beta was not set, so defaulting to beta = 1e20 "
                "which is so"
                "large that the ||Qijk + Qjik + Qkij|| "
                "term in the optimization will be essentially ignored."
            )
        if alpha_m < 0 or alpha_m > eta:
            raise ValueError("0 <= alpha_m <= eta")
        if alpha_A < 0 or alpha_A > eta:
            raise ValueError("0 <= alpha_A <= eta")
        if gamma >= 0:
            raise ValueError("gamma must be negative")
        if tol <= 0 or tol_m <= 0 or eps_solver <= 0:
            raise ValueError("tol and tol_m must be positive")

        self.mod_matrix = mod_matrix
        self.evolve_w = evolve_w
        self.eps_solver = eps_solver
        self.inequality_constraints = inequality_constraints
        self.m0 = m0
        self.A0 = A0
        self.alpha_A = alpha_A
        self.alpha_m = alpha_m
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tol = tol
        self.tol_m = tol_m
        self.accel = accel
        self.A_history_ = []
        self.m_history_ = []
        self.p_history_ = []
        self.PW_history_ = []
        self.PWeigs_history_ = []
        self.history_ = []
        self.objective_history = objective_history
        self.unbias = False
        self.verbose_cvxpy = verbose_cvxpy
        self.use_constraints = (constraint_lhs is not None) and (
            constraint_rhs is not None
        )
        if inequality_constraints:
            if not evolve_w:
                raise ValueError(
                    "Use of inequality constraints requires solving for xi "
                    " (evolve_w=True)."
                )
            if not self.use_constraints:
                raise ValueError(
                    "Use of inequality constraints requires constraint_rhs "
                    "and constraint_lhs "
                    "variables to be passed to the Optimizer class."
                )

        if self.use_constraints:
            if constraint_order not in ("feature", "target"):
                raise ValueError(
                    "constraint_order must be either 'feature' or 'target'"
                )

            self.constraint_lhs = constraint_lhs
            self.constraint_rhs = constraint_rhs
            self.unbias = False
            self.constraint_order = constraint_order

    @staticmethod
    def _build_PC(polyterms: list[tuple[int, Int1D]]) -> Float3D:
        r"""Build the matrix that projects out the constant term of a library

        Coefficients in each polynomial equation :math:`i\in \{1,\dots, r\}` can
        be stored in an array arranged as written out on paper (e.g.
        :math:` f_i(x) = a^i_0 + a^i_1 x_1, a^i_2 x_1x_2, \dots a^i_N x_r^2`) or
        in a series of matrices :math:`E \in \mathbb R^r`,
        :math:`L\in \mathbb R^{r\times r}`, and (without loss of generality) in
        :math:`Q\in \mathbb R^{r \times r \times r}, where each
        :math:`Q^{(i)}_{j,k}` is symmetric in the last two indexes.

        This function builds the projection tensor for extracting the constant
        terms :math:`E` from a set of coefficients in the first representation.

        Args:
            polyterms: the ordering and meaning of terms in the equations.  Each
                entry represents a term in the equation and comprises its index
                and an array of exponents for each variable

        Returns:
            3rd order tensor
        """
        n_targets, n_features, _, _, _ = _build_lib_info(polyterms)
        c_terms = [ind for ind, exps in polyterms if sum(exps) == 0]
        PC = np.zeros((n_targets, n_targets, n_features))
        if c_terms:  # either a length 0 or length 1 list
            PC[range(n_targets), range(n_targets), c_terms[0]] = 1.0
        return PC

    @staticmethod
    def _build_PL(polyterms: list[tuple[int, Int1D]]) -> tuple[Float4D, Float4D]:
        r"""Build the matrix that projects out the linear terms of a library

        Coefficients in each polynomial equation :math:`i\in \{1,\dots, r\}` can
        be stored in an array arranged as written out on paper (e.g.
        :math:` f_i(x) = a^i_0 + a^i_1 x_1, a^i_2 x_1x_2, \dots a^i_N x_r^2`) or
        in a series of matrices :math:`E \in \mathbb R^r`,
        :math:`L\in \mathbb R^{r\times r}`, and (without loss of generality) in
        :math:`Q\in \mathbb R^{r \times r \times r}, where each
        :math:`Q^{(i)}_{j,k}` is symmetric in the last two indexes.

        This function builds the projection tensor for extracting the linear
        terms :math:`L` from a set of coefficients in the first representation.
        The function also calculates the projection tensor for extracting the
        symmetrized version of L

        Args:
            polyterms: the ordering and meaning of terms in the equations.  Each
                entry represents a term in the equation and comprises its index
                and an array of exponents for each variable

        Returns:
            Two 4th order tensors, the first one symmetric in the first two
            indexes.
        """
        n_targets, n_features, lin_terms, _, _ = _build_lib_info(polyterms)
        PL_tensor_unsym = np.zeros((n_targets, n_targets, n_targets, n_features))
        tgts = range(n_targets)
        for j in range(n_targets):
            PL_tensor_unsym[tgts, j, tgts, lin_terms[j]] = 1
        PL_tensor = (PL_tensor_unsym + np.transpose(PL_tensor_unsym, [1, 0, 2, 3])) / 2
        return cast(Float4D, PL_tensor), cast(Float4D, PL_tensor_unsym)

    @staticmethod
    def _build_PQ(polyterms: list[tuple[int, Int1D]]) -> Float5D:
        r"""Build the matrix that projects out the quadratic terms of a library

        Coefficients in each polynomial equation :math:`i\in \{1,\dots, r\}` can
        be stored in an array arranged as written out on paper (e.g.
        :math:` f_i(x) = a^i_0 + a^i_1 x_1, a^i_2 x_1x_2, \dots a^i_N x_r^2`) or
        in a series of matrices :math:`E \in \mathbb R^r`,
        :math:`L\in \mathbb R^{r\times r}`, and (without loss of generality) in
        :math:`Q\in \mathbb R^{r \times r \times r}, where each
        :math:`Q^{(i)}_{j,k}` is symmetric in the last two indexes.

        This function builds the projection tensor for extracting the quadratic
        forms :math:`Q` from a set of coefficients in the first representation.

        Args:
            polyterms: the ordering and meaning of terms in the equations.  Each
                entry represents a term in the equation and comprises its index
                and an array of exponents for each variable

        Returns:
            5th order tensor symmetric in second and third indexes.
        """
        n_targets, n_features, _, pure_terms, mixed_terms = _build_lib_info(polyterms)
        PQ = np.zeros((n_targets, n_targets, n_targets, n_targets, n_features))
        tgts = range(n_targets)
        for j, k in product(*repeat(range(n_targets), 2)):
            if j == k:
                PQ[tgts, j, k, tgts, pure_terms[j]] = 1.0
            if j != k:
                PQ[tgts, j, k, tgts, mixed_terms[frozenset({j, k})]] = 1 / 2
        return cast(Float5D, PQ)

    def _set_Ptensors(
        self, n_targets: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Make the projection tensors used for the algorithm."""
        lib = PolynomialLibrary(2, include_bias=self._include_bias).fit(
            np.zeros((1, n_targets))
        )
        polyterms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]

        PC_tensor = self._build_PC(polyterms)
        PL_tensor, PL_tensor_unsym = self._build_PL(polyterms)
        PQ_tensor = self._build_PQ(polyterms)
        PT_tensor = PQ_tensor.transpose([1, 0, 2, 3, 4])
        # PM is the sum of PQ and PQ which projects out the sum of Qijk and Qjik
        PM_tensor = PQ_tensor + PT_tensor

        return PC_tensor, PL_tensor_unsym, PL_tensor, PQ_tensor, PT_tensor, PM_tensor

    def _update_coef_constraints(self, H, x_transpose_y, P_transpose_A, coef_sparse):
        """Solves the coefficient update analytically if threshold = 0"""
        g = x_transpose_y + P_transpose_A / self.eta
        inv1 = np.linalg.pinv(H, rcond=1e-15)
        inv2 = np.linalg.pinv(
            self.constraint_lhs.dot(inv1).dot(self.constraint_lhs.T), rcond=1e-15
        )

        rhs = g.flatten() + self.constraint_lhs.T.dot(inv2).dot(
            self.constraint_rhs - self.constraint_lhs.dot(inv1).dot(g.flatten())
        )
        rhs = rhs.reshape(g.shape)
        return inv1.dot(rhs)

    def _update_A(self, A_old, PW):
        """Update the symmetrized A matrix"""
        eigvals, eigvecs = np.linalg.eigh(A_old)
        eigPW, eigvecsPW = np.linalg.eigh(PW)
        r = A_old.shape[0]
        A = np.diag(eigvals)
        for i in range(r):
            if eigvals[i] > self.gamma:
                A[i, i] = self.gamma
        return eigvecsPW @ A @ np.linalg.inv(eigvecsPW)

    def _convergence_criterion(self):
        """Calculate the convergence criterion for the optimization over w"""
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        err_coef = np.sqrt(np.sum((this_coef - last_coef) ** 2))
        return err_coef

    def _m_convergence_criterion(self):
        """Calculate the convergence criterion for the optimization over m"""
        return np.sum(np.abs(self.m_history_[-2] - self.m_history_[-1]))

    def _objective(self, x, y, coef_sparse, A, PW, k):
        """Objective function"""
        # Compute the errors
        R2 = (y - np.dot(x, coef_sparse)) ** 2
        A2 = (A - PW) ** 2
        Qijk = np.tensordot(
            self.mod_matrix,
            np.tensordot(self.PQ_, coef_sparse, axes=([4, 3], [0, 1])),
            axes=([1], [0]),
        )
        beta2 = (
            Qijk + np.transpose(Qijk, [1, 2, 0]) + np.transpose(Qijk, [2, 0, 1])
        ) ** 2
        L1 = self.threshold * np.sum(np.abs(coef_sparse.flatten()))
        R2 = 0.5 * np.sum(R2)
        stability_term = 0.5 * np.sum(A2) / self.eta
        alpha_term = 0.5 * np.sum(Qijk**2) / self.alpha
        beta_term = 0.5 * np.sum(beta2) / self.beta

        # convoluted way to print every max_iter / 10 iterations
        if self.verbose and k % max(1, self.max_iter // 10) == 0:
            row = [
                k,
                R2,
                stability_term,
                L1,
                alpha_term,
                beta_term,
                R2 + stability_term + L1 + alpha_term + beta_term,
            ]
            if self.threshold == 0:
                if k % max(int(self.max_iter / 10.0), 1) == 0:
                    print(
                        "{0:5d} ... {1:8.3e} ... {2:8.3e} ... {3:8.2e}"
                        " ... {4:8.2e} ... {5:8.2e} ... {6:8.2e}".format(*row)
                    )
            else:
                print(
                    "{0:5d} ... {1:8.3e} ... {2:8.3e} ... {3:8.2e}"
                    " ... {4:8.2e} ... {5:8.2e} ... {6:8.2e}".format(*row)
                )
        return R2 + stability_term + L1 + alpha_term + beta_term

    def _solve_sparse_relax_and_split(self, r, N, x_expanded, y, Pmatrix, A, coef_prev):
        """Solve coefficient update with CVXPY if threshold != 0"""
        xi = cp.Variable(N * r)
        cost = cp.sum_squares(x_expanded @ xi - y.flatten())
        if self.thresholder.lower() == "l1":
            cost = cost + self.threshold * cp.norm1(xi)
        elif self.thresholder.lower() == "weighted_l1":
            cost = cost + cp.norm1(np.ravel(self.thresholds) @ xi)
        elif self.thresholder.lower() == "l2":
            cost = cost + self.threshold * cp.norm2(xi) ** 2
        elif self.thresholder.lower() == "weighted_l2":
            cost = cost + cp.norm2(np.ravel(self.thresholds) @ xi) ** 2
        cost = cost + cp.sum_squares(Pmatrix @ xi - A.flatten()) / self.eta

        # new terms minimizing quadratic piece ||P^Q @ xi||_2^2
        Q = np.reshape(self.PQ_, (r * r * r, N * r), "F")
        cost = cost + cp.sum_squares(Q @ xi) / self.alpha
        Q = np.reshape(self.PQ_, (r, r, r, N * r), "F")
        Q_ep = Q + np.transpose(Q, [1, 2, 0, 3]) + np.transpose(Q, [2, 0, 1, 3])
        Q_ep = np.reshape(Q_ep, (r * r * r, N * r), "F")
        cost = cost + cp.sum_squares(Q_ep @ xi) / self.beta

        # Constraints
        if self.use_constraints:
            if self.inequality_constraints:
                prob = cp.Problem(
                    cp.Minimize(cost),
                    [self.constraint_lhs @ xi <= self.constraint_rhs],
                )
            else:
                prob = cp.Problem(
                    cp.Minimize(cost),
                    [self.constraint_lhs @ xi == self.constraint_rhs],
                )
        else:
            prob = cp.Problem(cp.Minimize(cost))

        # default solver is OSQP here but switches to ECOS for L2
        try:
            prob.solve(
                eps_abs=self.eps_solver,
                eps_rel=self.eps_solver,
                verbose=self.verbose_cvxpy,
            )
        # Annoying error coming from L2 norm switching to use the ECOS
        # solver, which uses "max_iters" instead of "max_iter", and
        # similar semantic changes for the other variables.
        except TypeError:
            try:
                prob.solve(
                    abstol=self.eps_solver,
                    reltol=self.eps_solver,
                    verbose=self.verbose_cvxpy,
                )
            except cp.error.SolverError:
                print("Solver failed, setting coefs to zeros")
                xi.value = np.zeros(N * r)
        except cp.error.SolverError:
            print("Solver failed, setting coefs to zeros")
            xi.value = np.zeros(N * r)

        if xi.value is None:
            warnings.warn(
                "Infeasible solve, increase/decrease eta",
                ConvergenceWarning,
            )
            return None
        coef_sparse = (xi.value).reshape(coef_prev.shape)
        return coef_sparse

    def _solve_m_relax_and_split(self, r, N, m_prev, m, A, coef_sparse, tk_previous):
        """
        If using the relaxation formulation of trapping SINDy, solves the
        (m, A) algorithm update.
        """
        # prox-grad for (A, m)
        # Accelerated prox gradient descent
        if self.accel:
            tk = (1 + np.sqrt(1 + 4 * tk_previous**2)) / 2.0
            m_partial = m + (tk_previous - 1.0) / tk * (m - m_prev)
            tk_previous = tk
            mPM = np.tensordot(self.PM_, m_partial, axes=([2], [0]))
        else:
            mPM = np.tensordot(self.PM_, m, axes=([2], [0]))
        p = np.tensordot(self.mod_matrix, self.PL_ + mPM, axes=([1], [0]))
        PW = np.tensordot(p, coef_sparse, axes=([3, 2], [0, 1]))
        PMW = np.tensordot(self.PM_, coef_sparse, axes=([4, 3], [0, 1]))
        PMW = np.tensordot(self.mod_matrix, PMW, axes=([1], [0]))
        A_b = (A - PW) / self.eta
        PMT_PW = np.tensordot(PMW, A_b, axes=([2, 1], [0, 1]))
        if self.accel:
            m_new = m_partial - self.alpha_m * PMT_PW
        else:
            m_new = m_prev - self.alpha_m * PMT_PW
        m_current = m_new

        # Update A
        A_new = self._update_A(A - self.alpha_A * A_b, PW)
        return m_current, m_new, A_new, tk_previous

    def _solve_nonsparse_relax_and_split(self, H, xTy, P_transpose_A, coef_prev):
        """Update for the coefficients if threshold = 0."""
        if self.use_constraints:
            coef_sparse = self._update_coef_constraints(
                H, xTy, P_transpose_A, coef_prev
            ).reshape(coef_prev.shape)
        else:
            # Alan Kaptanoglu: removed cho factor calculation here
            # which has numerical issues in certain cases. Easier to chop
            # things using pinv, but gives dumb results sometimes.
            warnings.warn(
                "TrappingSR3._solve_nonsparse_relax_and_split using "
                "naive pinv() call here, be careful with rcond parameter."
            )
            Hinv = np.linalg.pinv(H, rcond=1e-15)
            coef_sparse = Hinv.dot(xTy + P_transpose_A / self.eta).reshape(
                coef_prev.shape
            )
        return coef_sparse

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the
        TrappingSR3 algorithm.
        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """

        n_samples, n_features = x.shape
        self.n_features = n_features
        r = y.shape[1]
        N = n_features  # int((r ** 2 + 3 * r) / 2.0)
        if N > int((r**2 + 3 * r) / 2.0):
            self._include_bias = True

        if self.mod_matrix is None:
            self.mod_matrix = np.eye(r)

        # Define PL, PQ, PT and PM tensors, only relevant if the stability term in
        # trapping SINDy is turned on.
        (
            self.PC_,
            self.PL_unsym_,
            self.PL_,
            self.PQ_,
            self.PT_,
            self.PM_,
        ) = self._set_Ptensors(r)

        # Set initial coefficients
        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(
                self.constraint_lhs, n_features, output_order="target"
            )
        coef_sparse: np.ndarray[tuple[NFeat, NTarget], AnyFloat] = self.coef_.T

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iter",
                "|y-Xw|^2",
                "|Pw-A|^2/eta",
                "|w|_1",
                "|Qijk|/a",
                "|Qijk+...|/b",
                "Total:",
            ]
            print(
                "{: >5} ... {: >8} ... {: >10} ... {: >5}"
                " ... {: >8} ... {: >10} ... {: >8}".format(*row)
            )

        # initial A
        if self.A0 is not None:
            A = self.A0
        elif np.any(self.PM_ != 0.0):
            A = np.diag(self.gamma * np.ones(r))
        else:
            A = np.diag(np.zeros(r))
        self.A_history_.append(A)

        # initial guess for m
        if self.m0 is not None:
            m = self.m0
        else:
            np.random.seed(1)
            m = (np.random.rand(r) - np.ones(r)) * 2
        self.m_history_.append(m)

        # Precompute some objects for optimization
        x_expanded = np.zeros((n_samples, r, n_features, r))
        for i in range(r):
            x_expanded[:, i, :, i] = x
        x_expanded = np.reshape(x_expanded, (n_samples * r, r * n_features))
        xTx = np.dot(x_expanded.T, x_expanded)
        xTy = np.dot(x_expanded.T, y.flatten())

        # if using acceleration
        tk_prev = 1
        m_prev = m

        # Begin optimization loop
        objective_history = []
        for k in range(self.max_iter):

            # update P tensor from the newest m
            mPM = np.tensordot(self.PM_, m, axes=([2], [0]))
            p = np.tensordot(self.mod_matrix, self.PL_ + mPM, axes=([1], [0]))
            Pmatrix = p.reshape(r * r, r * n_features)

            # update w
            coef_prev = coef_sparse
            if self.evolve_w:
                if (self.threshold > 0.0) or self.inequality_constraints:
                    coef_sparse = self._solve_sparse_relax_and_split(
                        r, n_features, x_expanded, y, Pmatrix, A, coef_prev
                    )
                else:
                    # if threshold = 0, there is analytic expression
                    # for the solve over the coefficients,
                    # which is coded up here separately
                    pTp = np.dot(Pmatrix.T, Pmatrix)
                    # notice reshaping PQ here requires fortran-ordering
                    PQ = np.tensordot(self.mod_matrix, self.PQ_, axes=([1], [0]))
                    PQ = np.reshape(PQ, (r * r * r, r * n_features), "F")
                    PQTPQ = np.dot(PQ.T, PQ)
                    PQ = np.reshape(self.PQ_, (r, r, r, r * n_features), "F")
                    PQ = np.tensordot(self.mod_matrix, PQ, axes=([1], [0]))
                    PQ_ep = (
                        PQ
                        + np.transpose(PQ, [1, 2, 0, 3])
                        + np.transpose(PQ, [2, 0, 1, 3])
                    )
                    PQ_ep = np.reshape(PQ_ep, (r * r * r, r * n_features), "F")
                    PQTPQ_ep = np.dot(PQ_ep.T, PQ_ep)
                    H = xTx + pTp / self.eta + PQTPQ / self.alpha + PQTPQ_ep / self.beta
                    P_transpose_A = np.dot(Pmatrix.T, A.flatten())
                    coef_sparse = self._solve_nonsparse_relax_and_split(
                        H, xTy, P_transpose_A, coef_prev
                    )

            # If problem over xi becomes infeasible, break out of the loop
            if coef_sparse is None:
                coef_sparse = coef_prev
                break

            # Now solve optimization for m and A
            m_prev, m, A, tk_prev = self._solve_m_relax_and_split(
                r, n_features, m_prev, m, A, coef_sparse, tk_prev
            )

            # If problem over m becomes infeasible, break out of the loop
            if m is None:
                m = m_prev
                break
            self.history_.append(coef_sparse.T)
            PW = np.tensordot(p, coef_sparse, axes=([3, 2], [0, 1]))

            # (m,A) update finished, append the result
            self.m_history_.append(m)
            self.A_history_.append(A)
            eigvals, eigvecs = np.linalg.eig(PW)
            self.PW_history_.append(PW)
            self.PWeigs_history_.append(np.sort(eigvals))
            mPM = np.tensordot(self.PM_, m, axes=([2], [0]))
            p = np.tensordot(self.mod_matrix, self.PL_ + mPM, axes=([1], [0]))
            self.p_history_.append(p)

            # update objective
            objective_history.append(self._objective(x, y, coef_sparse, A, PW, k))

            if (
                self._m_convergence_criterion() < self.tol_m
                and self._convergence_criterion() < self.tol
            ):
                # Could not (further) select important features
                break
        if k == self.max_iter - 1:
            warnings.warn(
                "TrappingSR3._reduce did not converge after {} iters.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )

        self.coef_ = coef_sparse.T
        self.objective_history = objective_history


def _make_constraints(n_tgts: int, **kwargs):
    """Create constraints for the Quadratic terms in TrappingSR3.

    These are the constraints from equation 5 of the Trapping SINDy paper.

    Args:
        n_tgts: number of coordinates or modes for which you're fitting an ODE.
        kwargs: Keyword arguments to PolynomialLibrary such as
            ``include_bias``.

    Returns:
        A tuple of the constraint zeros, and a constraint matrix to multiply
        by the coefficient matrix of Polynomial terms. Number of constraints is
        ``n_tgts + 2 * math.comb(n_tgts, 2) + math.comb(n_tgts, 3)``.
        Constraint matrix is of shape ``(n_constraint, n_feature, n_tgt)``.
        To get "feature" order constraints, use
        ``np.reshape(constraint_matrix, (n_constraints, -1))``.
        To get "target" order constraints, transpose axis 1 and 2 before
        reshaping.
    """
    lib = PolynomialLibrary(2, **kwargs).fit(np.zeros((1, n_tgts)))
    terms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    _, n_terms, linear_terms, pure_terms, mixed_terms = _build_lib_info(terms)
    # index of tgt -> index of its pure quadratic term
    pure_terms = {np.argmax(exps): t_ind for t_ind, exps in terms if max(exps) == 2}
    # two indexes of tgts -> index of their mixed quadratic term
    mixed_terms = {
        frozenset(np.argwhere(exponent == 1).flatten()): t_ind
        for t_ind, exponent in terms
        if max(exponent) == 1 and sum(exponent) == 2
    }
    constraint_mat = np.vstack(
        (
            _pure_constraints(n_tgts, n_terms, pure_terms),
            _antisymm_double_constraint(n_tgts, n_terms, pure_terms, mixed_terms),
            _antisymm_triple_constraints(n_tgts, n_terms, mixed_terms),
        )
    )

    return np.zeros(len(constraint_mat)), constraint_mat


def _pure_constraints(n_tgts: int, n_terms: int, pure_terms: dict[int, int]) -> Float2D:
    """Set constraints for coefficients adorning terms like a_i^3 = 0"""
    constraint_mat = np.zeros((n_tgts, n_terms, n_tgts))
    for constr_ind, (tgt_ind, term_ind) in zip(range(n_tgts), pure_terms.items()):
        constraint_mat[constr_ind, term_ind, tgt_ind] = 1.0
    return constraint_mat


def _antisymm_double_constraint(
    n_tgts: int,
    n_terms: int,
    pure_terms: dict[int, int],
    mixed_terms: dict[frozenset[int], int],
) -> Float2D:
    """Set constraints for coefficients adorning terms like a_i^2 * a_j=0"""
    constraint_mat_1 = np.zeros((comb(n_tgts, 2), n_terms, n_tgts))  # a_i^2 * a_j
    constraint_mat_2 = np.zeros((comb(n_tgts, 2), n_terms, n_tgts))  # a_i * a_j^2
    for constr_ind, ((tgt_i, tgt_j), mix_term) in zip(
        range(n_tgts), mixed_terms.items()
    ):
        constraint_mat_1[constr_ind, mix_term, tgt_i] = 1.0
        constraint_mat_1[constr_ind, pure_terms[tgt_i], tgt_j] = 1.0
        constraint_mat_2[constr_ind, mix_term, tgt_j] = 1.0
        constraint_mat_2[constr_ind, pure_terms[tgt_j], tgt_i] = 1.0

    return np.concatenate((constraint_mat_1, constraint_mat_2), axis=0)


def _antisymm_triple_constraints(
    n_tgts: int, n_terms: int, mixed_terms: dict[frozenset[int], int]
) -> Float2D:
    constraint_mat = np.zeros((comb(n_tgts, 3), n_terms, n_tgts))  # a_ia_ja_k

    def find_symm_term(a, b):
        return mixed_terms[frozenset({a, b})]

    for constr_ind, (tgt_i, tgt_j, tgt_k) in enumerate(combo_nr(range(n_tgts), 3)):
        constraint_mat[constr_ind, find_symm_term(tgt_j, tgt_k), tgt_i] = 1
        constraint_mat[constr_ind, find_symm_term(tgt_k, tgt_i), tgt_j] = 1
        constraint_mat[constr_ind, find_symm_term(tgt_i, tgt_j), tgt_k] = 1

    return constraint_mat


def _build_lib_info(
    polyterms: list[tuple[int, Int1D]]
) -> tuple[int, int, dict[int, int], dict[int, int], dict[frozenset[int], int]]:
    """From polynomial, calculate various useful info

    Args:
        polyterms.  The output of PolynomialLibrary.powers_.  Each term is
            a tuple of it's index in the ordering and a 1D array of the
            exponents of each feature.

    Returns:
        the number of targets
        the number of features
        a dictionary from each target to its linear term index
        a dictionary from each target to its quadratic term index
        a dictionary from each pair of targets to its mixed term index
    """
    try:
        n_targets = len(polyterms[0][1])
    except IndexError:
        raise ValueError("Passed a polynomial library with no terms")
    n_features = len(polyterms)
    mixed_terms = {
        frozenset(np.argwhere(exps == 1).flatten()): t_ind
        for t_ind, exps in polyterms
        if max(exps) == 1 and sum(exps) == 2
    }
    pure_terms = {np.argmax(exps): t_ind for t_ind, exps in polyterms if max(exps) == 2}
    linear_terms = {
        np.argmax(exps): t_ind for t_ind, exps in polyterms if sum(exps) == 1
    }
    return n_targets, n_features, linear_terms, pure_terms, mixed_terms
