""":math:`GL^+`: Positive Component of the Generalized Linear group with left-invariant metric.
"""

import jax.numpy.linalg as jla
from .matrix_left_invariant import MatrixLeftInvariant
from ..utils.utils import (grand)


class GLpLeftInvariant(MatrixLeftInvariant):
    """:math:`GL^+` with left invariant metric defined by g_mat.

    :param p: the size of the matrix
    :param g_mat: The matrix defining the inner product at the identity. g_mat is in :math:`\\mathbb{R}^{p^2\\times p^2}` .

    """
    def name(self):
        return f"GL+({self.shape[0]})"

    def _mat_apply(self, mat, omg):
        return (mat@omg.reshape(-1)).reshape(self.shape)

    def _lie_algebra_proj(self, omg):
        return omg

    def rand_ambient(self, key):
        """random ambient vector
        """
        return grand(key, (self.shape))

    def rand_point(self, key):
        """ A random point on the manifold
        """
        ret, key = self.rand_ambient(key)
        if jla.det(ret) < 0:
            return ret.at[0, :].set(-ret[0, :]), key
        return ret, key

    def retract(self, x, v):
        """ second order retraction, but simple
        """
        return x + v - 0.5* self.proj(x, self.gamma(x, v, v))

    def approx_nearest(self, q):
        return q

    def pseudo_transport(self, x, y, v):
        """the easy one
        """
        return y@jla.solve(x, v)

    def sigma(self, x, dw):
        """ sigma is applied on a vector rather than a matrix
        """
        return x@self._mat_apply(self._i_sqrt_g_mat, dw)
