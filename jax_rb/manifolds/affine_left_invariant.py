""":math:`Aff^+`: Positive Component of the Affine group with left invariant metric.
"""

import jax.numpy.linalg as jla
from .matrix_left_invariant import MatrixLeftInvariant
from ..utils.utils import (grand)


class AffineLeftInvariant(MatrixLeftInvariant):
    """Group of affine tranformations of :math:`\\mathbb{R}^{n}`,
    represented by a pair :math:`(A, v)\\in GL^+(n)\\times \\mathbb{R}^{n}`
    with action :math:`(A, v).w = Aw + v` for :math:`w\\in\\mathbb{R}^{n}` .

    Alternatively, it is represented as a matrix :math:`\\begin{bmatrix} A & v \\\\ 0 & 1 \\end{bmatrix}\\in  GL(n+1)`.

    :param n: size of A
    :param g_mat: a positive definite matrix in :math:`\\mathbb{R}^{n(n+1)\\times n(n+1)}` defining the metric at :math:`I_{n+1}`
    """
    def __init__(self, n, g_mat):
        """ g_mat is a matrix of size (n(n+1))**2
        used to define the metric
        """
        super().__init__(n+1, g_mat)
        self.dim = n*(n+1)

    def name(self):
        return f"Aff({self.shape[0]-1})"

    def _lie_algebra_proj(self, omg):
        """ The projection at identity
        """
        return omg.at[-1, :].set(0.)

    def _mat_apply(self, mat, omg):
        """ mat is a matrix of size (p(p-1))**2        
        """
        p = omg.shape[0]
        return omg.at[:-1, :].set(
            (mat@omg[:-1, :].reshape(-1)).reshape(p-1, p))

    def rand_point(self, key):
        """ A random point on the manifold
        """
        mat, key = grand(key, self.shape)
        return mat.at[-1, :].set(0.).at[-1, -1].set(1.), key

    def retract(self, x, v):
        """ second order retraction, but simple
        """
        return (x + v -0.5*self.gamma(x, v, v)).at[-1, :].set(0.)

    def approx_nearest(self, q):
        return q.at[-1, :].set(0.)

    def pseudo_transport(self, x, y, v):
        """the easy one
        """
        return y@jla.solve(x, v)

    def sigma(self, x, dw):
        return x@self._lie_algebra_proj(self._mat_apply(self._i_sqrt_g_mat, dw))
