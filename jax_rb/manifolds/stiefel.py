"""Stiefel manifold :math:`\\mathrm{St}(n, p, \\alpha_0, \\alpha_1)` with metric defined by two parameters.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax.scipy.linalg import expm
from ..utils.utils import (grand, sym, asym, esqrtm, unvec_skew)
from .global_manifold import GlobalManifold


class RealStiefelAlpha(GlobalManifold):
    """The manifold :math:`Y^TY = I` where :math:`Y` is a matrix of size :math:`shape=n\\times p`    with metric :math:`\\lvert \\omega\\rvert^2_{\\mathsf{g}} =\\alpha_0 Tr(\\omega^T\\omega) +(\\alpha_0-\\alpha_1)Tr(\\omega^TYY^T\\omega)` .

    :param shape: tuple (n, p),
    :param alpha: array of 2 positive numbers.
    """
    def __init__(self, shape, alpha):
        """Constructor
        """
        self.shape = shape
        n, p = shape
        self.dim = (n-p)*p+(p*(p-1))//2
        self.alpha = alpha

    def name(self):
        return f"Stiefel({self.shape}) alpha={self.alpha}"

    def inner(self, x, a, b):
        al = self.alpha
        return al[0]*jnp.sum(a*b) + (al[1]-al[0])*jnp.sum((x.T@a)*(x.T@b))

    def g_metric(self, x, omg):
        """ the metric operator g
        """
        al = self.alpha
        return al[0]*omg + (al[1]-al[0])*x@(x.T@omg)

    def inv_g_metric(self, x, omg):
        """ inverse of the metric operator g
        """
        al = self.alpha
        return 1/al[0]*omg + (1/al[1]-1/al[0])*x@(x.T@omg)

    @partial(jax.jit, static_argnums=(0,))
    def proj(self, x, omg):
        """ Metric compatible projection
        """
        return omg - x@sym(x.T@omg)

    def rand_vec(self, key, x):
        """random tangent vector
        """
        omg, key = grand(key, self.shape)
        return self.proj(x, omg), key

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

        def grass_proj(omg):
            return omg - x@(x.T@omg)

        al = self.alpha
        return x@sym(xi.T@eta) \
            + (al[0] - al[1])/al[0]*grass_proj(xi@(eta.T@x) + eta@(xi.T@x))

    @partial(jax.jit, static_argnums=(0,))
    def retract(self, x, v):
        """ second order retraction, but simple
        """
        x1 = x+ v - 0.5* self.proj(x, self.gamma(x, v, v))
        ei, ev = jla.eigh(x1.T@x1)
        return x1@ev@((1/jnp.sqrt(ei))[:, None]*ev.T)

    @partial(jax.jit, static_argnums=(0,))
    def approx_nearest(self, q):
        """ second order retraction, but simple
        """
        # return jax.scipy.linalg.polar(q)[0]
        ei, ev = jla.eigh(q.T@q)
        return q@ev@((1/jnp.sqrt(ei))[:, None]*ev.T)

    def gamma_ambient(self, x, omg1, omg2):
        """ gamma of the metric on the ambient space
        """
        al = self.alpha
        return (al[1]-al[0])*self.inv_g_metric(
            x,
            omg1@asym(x.T@omg2)
            + x@sym(omg1.T@omg2)
            + omg2@asym(x.T@omg1)
        )

    @partial(jax.jit, static_argnums=(0,))
    def ito_drift(self, x):
        al = self.alpha
        n, p = self.shape
        return -0.5*((n-p)/al[0] + 0.5*(p-1)/al[1])*x

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
        alh = 1/jnp.sqrt(self.alpha)
        return alh[0]*dw + (alh[1]-alh[0])*x@(x.T@dw)

    def sigma0(self, x, dw0):
        """ dw is a vector space of size self.dim.
        We use this for the geodesic
        walk strategy. Need storage for the complement
        """
        alh = 1/jnp.sqrt(self.alpha)
        n, p = self.shape
        # ret = alh[1]*x@unvecah(dw0[((p*(p-1)))//2])
        pk = ((p*(p-1)))//2
        ret = alh[1]*x@unvec_skew(dw0[:pk])
        P = esqrtm(x[:p, :].T@x[:p, :])
        Q = jla.solve(P, x[:p, :].T).T
        B = dw0[pk:].reshape(n-p, p)
        ret += alh[0]*jnp.concatenate(
            [- Q@x[p:, :].T@B,
             B - x[p:, :]@jla.solve(P + jnp.eye(p), x[p:, :].T@B)],
            axis=0)
        return ret

    def pseudo_transport(self, x, y, v):
        v1 = self.proj(y, v)
        return v1/jnp.sqrt(self.inner(y, v1, v1))

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
        K = eta - x @ (x.T @ eta)
        xp, R = jla.qr(K)
        alf = self.alpha[1]/self.alpha[0]
        A = x.T @ eta
        x_mat = jnp.concatenate([
            jnp.concatenate([2*alf*A, -R.T], axis=1),
            jnp.concatenate([R, jnp.zeros((p, p))], axis=1)], axis=0)
        return jnp.array(
            jnp.concatenate([x, xp], axis=1) @ expm(x_mat)[:, :p] @ expm((1-2*alf)*A))
