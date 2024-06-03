""":math:`SE`: Special Euclidean group.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from .matrix_left_invariant import MatrixLeftInvariant
from ..utils.utils import (sym, asym)


class SELeftInvariant(MatrixLeftInvariant):
    """Specian Euclidean group of Euclidean transformation of :math:`\\mathbb{R}^{n}`,
    represented by a pair :math:`(A, v)\\in SO^(n)\\times \\mathbb{R}^{n}`
    with action :math:`(A, v).w = Aw + v` for :math:`w\\in\\mathbb{R}^{n}` .

    Alternatively, it is represented as a matrix :math:`\\begin{bmatrix} A & v \\\\ 0 & 1 \\end{bmatrix}\\in  GL(n+1)` where :math:`A\\in SO(n)`.

    :param n: size of A
    :param g_mat: a positive definite matrix in :math:`\\mathbb{R}^{\\frac{n(n+1)}{2}\\times\\frac{n(n+1)}{2}}` defining the metric at :math:`I_{n+1}`
    """
    def __init__(self, n, g_mat):
        """ g_mat is a matrix of size (n(n+1)//2)**2
        used to define the metric
        """
        super().__init__(n+1, g_mat)
        self.dim = (n*(n+1))//2

    def name(self):
        return f"SE({self.shape[0]-1})"

    @partial(jax.jit, static_argnums=(0,))
    def _lie_algebra_proj(self, omg):
        return omg.at[:-1, :-1].set(asym(omg[:-1, :-1])).at[-1, :].set(0.)

    def _mat_apply(self, mat, omg):
        """ mat is a matrix of size (p(p+1)/)**2
        Trick is to vectorize both the symmetric
        and anti symmetric away from the diagonal
        """
        p = omg.shape[0]
        sqrt2 = jnp.sqrt(2)

        reta = asym(omg[:-1, :-1])
        rets = sym(omg[:-1, :-1])

        rows, cols = jnp.triu_indices(p-1, 1)
        veca = mat@jnp.concatenate([reta.take(rows*(p-1)+cols)*sqrt2,
                                    omg[:-1, -1]])
        vecs = mat@jnp.concatenate([rets.take(rows*(p-1)+cols)*sqrt2,
                                    omg[-1, :-1]])

        ret = jnp.empty((p, p))
        ret = ret.at[:-1, -1].set(veca[1-p:])
        ret = ret.at[-1, :-1].set(vecs[1-p:])

        tota = jnp.zeros((p-1, p-1))
        tota = tota.at[rows, cols].set(veca[:1-p])
        tota = tota.at[cols, rows].set(-veca[:1-p])

        tots = jnp.zeros((p-1, p-1))
        tots = tots.at[rows, cols].set(vecs[:1-p])
        tots = tots.at[cols, rows].set(vecs[:1-p])

        ret = ret.at[:-1, :-1].set((tota+tots)/sqrt2)
        return ret.at[jnp.diag_indices(p)].set(omg[jnp.diag_indices(p)])

    def rand_point(self, key):
        """ A random point on the manifold
        """
        xt, key = self.rand_ambient(key)
        xo = jla.qr(xt[:-1, :-1])[0]
        if jla.det(xo) < 0:
            xo = xo.at[0, :].set(-xo[0, :])

        return xt.at[:-1, :-1].set(xo).at[-1, :-1].set(0.).at[-1, -1].set(1.), key

    # @partial(jax.jit, static_argnums=(0,))
    def retract(self, x, v):
        """ second order retraction, but simple
        """
        x1 = x + v - 0.5* self.proj(x, self.gamma(x, v, v))
        u, _, v = jla.svd(x1[:-1, :-1])
        return x1.at[:-1, :-1].set(u@v)

    @partial(jax.jit, static_argnums=(0,))    
    def approx_nearest(self, q):
        u, _, v = jla.svd(q[:-1, :-1])
        return q.at[:-1, :-1].set(u@v)

    def pseudo_transport(self, x, y, v):
        """the easy one
        """
        return y@x.T@v

    def sigma(self, x, dw):
        return x@self._lie_algebra_proj(self._mat_apply(self._i_sqrt_g_mat, dw))
