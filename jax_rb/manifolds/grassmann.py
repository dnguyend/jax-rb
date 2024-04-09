"""Grassmann manifold :math:`\\mathrm{Gr}(n, p)` of vector spaces of rank :math:`p` in a :math:`n` -dimension vector space.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax.scipy.linalg import expm
from ..utils.utils import (esqrtm)
from .global_manifold import GlobalManifold



class Grassmann(GlobalManifold):
    """ 
    The lift of the Grassman manifold to to the Stiefel manifold :math:`Y^TY = I` where :math:`Y` is a matrix of :math:`shape=n\\times p` with metric :math:`\\lvert \\omega\\rvert^2_{\\mathsf{g}} = Tr(\\omega^T\\omega)`. The lift is with respect to the submersion defined by the relationship :math:`Y\\sim YU` for an orthogonal matrix :math:`U`.
    """
    def __init__(self, shape):
        """Constructor
        """
        self.shape = shape
        n, p = shape
        self.dim = (n-p)*p

    def name(self):
        return f"Gr({self.shape[0]}, {self.shape[1]})"

    def inner(self, x, a, b):
        return jnp.sum(a*b)

    def g_metric(self, x, omg):
        """ the metric operator g
        """
        return omg

    def inv_g_metric(self, x, omg):
        """ inverse of the metric operator g
        """
        return omg

    @partial(jax.jit, static_argnums=(0,))
    def proj(self, x, omg):
        """ Metric compatible projection
        """
        return omg - x@x.T@omg

    def rand_point(self, key):
        """ A random point on the manifold
        """
        xt, key = self.rand_ambient(key)
        x, _ = jla.qr(xt)
        return x, key

    @partial(jax.jit, static_argnums=(0,))
    def gamma(self, x, xi, eta):
        """Christoffel function
        """
        return x@(xi.T@eta)

    @partial(jax.jit, static_argnums=(0,))    
    def retract(self, x, v):
        """ second order retraction, but simple
        """
        x1 = x+ v
        ei, ev = jla.eigh(x1.T@x1)
        return x1@ev@((1/jnp.sqrt(ei))[:, None]*ev.T)

    @partial(jax.jit, static_argnums=(0,))
    def approx_nearest(self, q):
        """ second order retraction, but simple
        """
        # return jax.scipy.linalg.polar(q)[0]
        ei, ev = jla.eigh(q.T@q)
        return q@ev@((1/jnp.sqrt(ei))[:, None]*ev.T)

    @partial(jax.jit, static_argnums=(0,))
    def ito_drift(self, x):
        n, p = self.shape
        return -0.5*(n-p)*x

    @partial(jax.jit, static_argnums=(0,))
    def stratonovic_drift(self, _):
        """ stratnovich drift
        """
        return jnp.zeros_like(self.shape)

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

    def sigma(self, x, dw):
        return dw

    def sigma0(self, x, dw0):
        """ dw is a vector space of size self.dim.
        We use this for the geodesic
        walk strategy. Need storage for the complement
        """
        n, p = self.shape
        ret = jnp.zeros_like(x)
        P = esqrtm(x[:p, :].T@x[:p, :])
        Q = jla.solve(P, x[:p, :].T).T
        B = dw0.reshape(n-p, p)
        ret += jnp.concatenate(
            [- Q@x[p:, :].T@B,
             B - x[p:, :]@jla.solve(P + jnp.eye(p), x[p:, :].T@B)],
            axis=0)
        return ret

    def pseudo_transport(self, x, y, v):
        """This is the actual parallel transport
        """
        # v1 = self.proj(y, v)
        # return v1/jnp.sqrt(self.inner(y, v1, v1))
        u, s, w = jla.svd(x.T@y)
        return (- x@u - y@w.T)@jnp.diag(1/(1+s))@w@(y.T@v) + v

    def exp(self, x, eta):
        """ Geodesics, the formula involves matrices of size 2d

        Parameters
        ----------
        x    : a manifold point
        eta  : tangent vector
        
        Returns
        ----------
        gamma(1), where gamma(t) is the geodesics at Y in direction eta

        """
        p = eta.shape[1]
        xp, r = jla.qr(eta)
        x_mat = jnp.concatenate([
            jnp.concatenate([jnp.zeros((p, p)), -r.T], axis=1),
            jnp.concatenate([r, jnp.zeros((p, p))], axis=1)], axis=0)
        return jnp.concatenate([x, xp], axis=1) @ expm(x_mat)[:, :p]
