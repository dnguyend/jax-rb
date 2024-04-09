"""Symmetric Positive Definite Matrix Manifold :math:`\\mathrm{S}^+(p)` of positive definite matrices of shape :math:`p\\times p` .
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax.scipy.linalg import expm

from ..utils.utils import (sym)
from .global_manifold import GlobalManifold


class PositiveDefiniteManifold(GlobalManifold):
    """The manifold of positive definite matrices of size :math:`p` .
    """
    def __init__(self, p):
        """Constructor
        """
        self.shape = (p, p)
        self.dim = (p*(p+1))//2

    def name(self):
        return f"S+({self.shape[0]})"

    def inner(self, x, a, b):
        return jnp.sum(jla.solve(x, a)*jla.solve(x, b.T).T)

    def g_metric(self, x, omg):
        return jla.solve(x, jla.solve(x, omg.T).T)

    def inv_g_metric(self, x, omg):
        return x@omg@x

    @partial(jax.jit, static_argnums=(0,))
    def proj(self, x, omg):
        return sym(omg)

    def rand_point(self, key):
        xt, key = self.rand_ambient(key)
        return sym(xt@xt.T), key

    @partial(jax.jit, static_argnums=(0,))
    def gamma(self, x, xi, eta):
        return -sym(xi@jla.solve(x, eta))

    def retract(self, x, v):
        return x+ v - 0.5* self.proj(x, self.gamma(x, v, v))

    def approx_nearest(self, q):
        """ point on the manifold nearest to q.
        """
        return q

    def gamma_ambient(self, x, omg1, omg2):
        """ gamma on the ambient space.
        """
        return -sym(omg1@jla.solve(x, omg2))

    @partial(jax.jit, static_argnums=(0,))
    def ito_drift(self, x):
        return (self.shape[0]+1)/4*x

    @partial(jax.jit, static_argnums=(0,))
    def stratonovich_drift(self, x):
        """ Brownian Stratonovich drift
        """
        ei, ev = jla.eigh(x)
        sei = jnp.sqrt(jnp.abs(ei))
        ss = -jnp.array([sei[i]**2*(.5+jnp.sum(sei/(sei[i]+sei)))
                         for i in range(self.shape[0])])
        return 0.5*ev@(ss[:, None]*ev.T) + (self.shape[0]+1)/4*x

    def laplace_beltrami(self, x, egradx, ehessvp):
        n, p = self.shape
        tup = jnp.zeros(self.shape)
        ret = 0
        for i in range(n):
            for j in range(p):
                e_ij = tup.at[i, j].set(1.)
                ret += self.proj(x, self.inv_g_metric(
                    x, ehessvp(x, e_ij)))[i, j]
        return ret + 2*jnp.sum(self.ito_drift(x)*egradx)

    def pseudo_transport(self, x, y, v):
        return v

    def sigma(self, x, dw):
        def xhf(x):
            ei, ev = jla.eigh(x)
            return ev@(jnp.sqrt(ei)[:, None]*ev.T)

        x2 = xhf(x)
        # return x2@unvech(dw)@x2
        return x2@sym(dw)@x2

    def exp(self, x, v):
        """ Geodesic."""
        return sym(x@expm(jla.solve(x, v)))
