""":math:`SL`: special linear group of matrices of determinant 1.
"""

import jax.numpy as jnp
import jax.numpy.linalg as jla
from .matrix_left_invariant import MatrixLeftInvariant
from ..utils.utils import (grand, complement_basis_for_vector)


class SLLeftInvariant(MatrixLeftInvariant):
    """:math:`SL(n)` Special linear group, of size :math:`n` with left invariant metric
    defined by g_mat.

    :param n: size of the matrix
    :param g_mat: a positive definite matrix in :math:`\\mathbb{R}^{(n^2-1)\\times (n^2-1)}` defining the metric at :math:`I_{n}`
    """
    def __init__(self, n, g_mat):
        """ g_mat is a matrix of size (n**2-1))**2
        used to define the metric
        """
        normal = 1/jnp.sqrt(n)*jnp.ones(n)
        self._u = jnp.concatenate(
            [complement_basis_for_vector(normal), normal[:, None]], axis=1)

        super().__init__(n, g_mat)
        self.dim = n*n-1

    def name(self):
        return f"SL({self.shape[0]})"

    def _vec_sl(self, mat):
        """ vectorize an n time n matrix
        with structure 
           upper -> first n(n-1)/2 
           lower -> next n(n-1)/2
           diagonal 
        """
        n = mat.shape[0]
        ret = jnp.empty(n*n)
        n_up = n*(n-1)//2
        ret = ret.at[:n_up].set(mat[jnp.triu_indices(n, 1)])
        ret = ret.at[n_up:2*n_up].set(mat[jnp.tril_indices(n, -1)])
        return ret.at[2*n_up:].set(self._u.T@jnp.diagonal(mat))

    def _unvec_sl(self, v):
        """ unravel a n*n vector to a trace 0 vector
        with 
        """
        n = self.shape[0]
        ret = jnp.empty((n, n))
        n_up = n*(n-1)//2
        ret = ret.at[jnp.triu_indices(n, 1)].set(v[:n_up])
        ret = ret.at[jnp.tril_indices(n, -1)].set(v[n_up:2*n_up])
        return ret.at[jnp.diag_indices(n)].set(self._u@v[2*n_up:])

    def _mat_apply(self, mat, omg):
        """ mat is of size (n**-1)**2
        multiply on all cells except for last one
        """
        # return (omg.reshape(-1).at[:-1].set(mat@omg.reshape(-1)[:-1])).reshape(self.shape)
        v = self._vec_sl(omg)
        return self._unvec_sl(v.at[:-1].set(mat@v[:-1]))

    def _lie_algebra_proj(self, omg):
        return omg - jnp.diag(jnp.full((self.shape[0]), jnp.trace(omg)/self.shape[0]))

    def rand_ambient(self, key):
        """random ambient vector
        """
        return grand(key, (self.shape))

    def rand_point(self, key):
        """ A random point on the manifold
        """
        ret, key = self.rand_ambient(key)
        if jla.det(ret) < 0:
            return self.approx_nearest(ret.at[0, :].set(-ret[0, :])), key
        return self.approx_nearest(ret), key

    def retract(self, x, v):
        """ second order retraction, but simple
        """
        return self.approx_nearest(x + v - 0.5* self.proj(x, self.gamma(x, v, v)))

    def approx_nearest(self, q):
        return q/jla.det(q)**(1/self.shape[0])

    def pseudo_transport(self, x, y, v):
        """the easy one
        """
        return y@jla.solve(x, v)

    def sigma(self, x, dw):
        """ sigma is applied on a vector rather than a matrix
        """
        return x@self._mat_apply(self._i_sqrt_g_mat, dw)
