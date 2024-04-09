""":math:`SO`: Special Orthogonal group.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from .matrix_left_invariant import MatrixLeftInvariant
from ..utils.utils import (asym, lie, jpolar)


class SOLeftInvariant(MatrixLeftInvariant):
    """The group :math:`SO(n)` of orthogonal matrices :math:`U\\in R^{n\\times n}`
    of determinant 1 with metric :math:`Tr(\\omega^TU^T\\mathcal{I}(U^T\\omega))`
    where :math:`\\mathcal{I}` is the metric defined by g_mat, a matrix of size
    :math:`\\mathbb{R}^{\\frac{n(n-1)}{2}\\times \\frac{n(n-1)}{2}}`.
    """
    def __init__(self, n, g_mat):
        """ g_mat is a matrix of size :math:`\\frac{n(n-1)}{2}`
        used to define the metric.
        """
        super().__init__(n, g_mat)
        self.dim = (n*(n-1)) // 2

    def name(self):
        return f"SO({self.shape[0]})"

    @partial(jax.jit, static_argnums=(0,))
    def _lie_algebra_proj(self, omg):
        return asym(omg)

    @partial(jax.jit, static_argnums=(0,))
    def _mat_apply(self, mat, omg):
        """ mat is a matrix of size (p(p-1))**2        
        """
        p = omg.shape[0]
        rows, cols = jnp.triu_indices(p, 1)

        ret = jnp.empty((p, p))
        ret = ret.at[rows, cols].set(mat@omg.take(rows*p+cols))
        ret = ret.at[cols, rows].set(mat@omg.T.take(rows*p+cols))
        ret = ret.at[jnp.diag_indices(p)].set(omg[jnp.diag_indices(p)])

        return ret

    def rand_point(self, key):
        """ A random point on the manifold
        """
        xt, key = self.rand_ambient(key)
        x, _ = jla.qr(xt)
        return x, key

    # @partial(jax.jit, static_argnums=(0,))
    def retract(self, x, v):
        """ second order retraction, but simple
        """
        x1 = x + v - 0.5* self.proj_gamma(x, v, v)
        # return jax.scipy.linalg.polar(x1)[0]
        return jpolar(x1)

    # @partial(jax.jit, static_argnums=(0,))
    def approx_nearest(self, q):
        # return jax.scipy.linalg.polar(q)[0]
        return jpolar(q)

    # @partial(jax.jit, static_argnums=(0,))
    def pseudo_transport(self, x, y, v):
        """the easy one
        """
        return y@x.T@v

    @partial(jax.jit, static_argnums=(0,))
    def proj_gamma(self, x, xi, eta):
        """projection of christoffel function
        """
        # return - self.d_proj(x, xi, eta)
        #    + self.proj(x, self.gamma_ambient(x, xi, eta))
        ivxi = jla.solve(x, xi)
        iveta = jla.solve(x, eta)

        return -0.5*x@self._lie_algebra_proj(ivxi@iveta + iveta@ivxi) \
            + 0.5*x@self._inv_id_opt(
                self._lie_algebra_proj(
                    lie(self._id_opt(ivxi), iveta.T) \
                    + lie(self._id_opt(iveta), ivxi.T)))

    # @partial(jax.jit, static_argnums=(0,))
    def sigma_la(self, vec_dw):
        """ sigma is applied on the lie agebra identified with a vector
        """
        p = self.shape[0]
        v = jnp.zeros(self.shape)
        rows, cols = jnp.triu_indices(p, 1)
        for idx in range(vec_dw.shape[0]):
            i, j = rows[idx], cols[idx]
            v += 1/jnp.sqrt(2)*vec_dw[idx]*self._mat_apply(
                self._i_sqrt_g_mat,
                jnp.zeros((p, p)).at[i, j].set(1.).at[j, i].set(-1.)
                )
        return v
