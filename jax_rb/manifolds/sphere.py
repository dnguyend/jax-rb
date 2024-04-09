""" Sphere of constant curvature
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from ..utils.utils import (grand, complement_basis_for_vector)
from .global_manifold import GlobalManifold


class Sphere(GlobalManifold):
    """The sphere :math:`x_0^2+x_1^2+\\cdots+x_d^2 = r^2`
    with sectional curvature :math:`\\frac{1}{r^2}`.

    :param n: dimension of :math:`\\mathcal{E}`
    :param r: the radius.    
    """
    def __init__(self, n, r):
        """
        """
        self.r = r
        self.dim = n-1
        self.shape = (n,)
        self.x0 = jnp.array([r] + (n-1)*[0.])

    def name(self):
        return f"Sphere S^{self.dim}"

    def inner(self, x, a, b):
        return jnp.sum(a*b)

    def g_metric(self, x, omg):
        return omg

    def inv_g_metric(self, x, omg):
        return omg

    def dist(self, x, y):
        """ distance between x and y
        """
        return self.r*jnp.arccos(1/self.r**2*jnp.sum(x*y))

    def log(self, x, y):
        """ Riemannian logarithm
        """
        al = 1/self.r**2*jnp.sum(x*y)
        return jnp.arccos(al)/(1-al**2)**.5*(y-al*x)

    def exp(self, x, v):
        """ geodesic
        """
        vnr = 1/self.r*jnp.sqrt(jnp.sum(v*v))
        ret = jnp.cos(vnr)*x + jnp.sin(vnr)*v/vnr
        return ret/jnp.sqrt(jnp.sum(ret*ret))*self.r

    @partial(jax.jit, static_argnums=(0,))
    def geodesic(self, x, v, t):
        """ geodesic
        """
        return self.exp(x, t*v)

    def d_exp(self, x, v, t):
        """ derivative in t of Exp(x, tv)
        """
        vnr = 1/self.r*jnp.sqrt(self.inner(x, v, v))
        return -jnp.sin(t*vnr)*vnr*x + jnp.cos(t*vnr)*v

    def _d2_exp(self, x, v, t):
        """ second derivativederivative in t of Exp(x, tv)
        """
        vnr = 1/self.r*jnp.sqrt(jnp.sum(v*v))
        # return -jnp.cos(t*vnr)*vnr**2*x - jnp.sin(t*vnr)/vnr*v
        return -jnp.cos(t*vnr)*vnr**2*x - jnp.sin(t*vnr)*vnr*v

    def rand_point(self, key):
        xt, key = grand(key, (self.shape[0],))
        return xt/jla.norm(xt, 2)*self.r, key

    def proj(self, x, omg):
        return omg - 1/self.r**2*x*jnp.sum(x*omg)

    def pseudo_transport(self, x, y, v):
        """ This is the real transport
        """
        return v - self.inner(x, y, v)/(self.r**2+self.inner(x, x, y))*(x+y)

    def transport_along_geodesic(self, x, v1, v, t):
        """ optimal transport along geodesic
        """
        vnr1 = 1/self.r*jnp.sqrt(jnp.sum(v1*v1))
        return v - jnp.sin(t*vnr1)/vnr1*self.inner(x, v1, v)/self.r**2*x \
            - 1/self.r**2*jnp.sin(t*vnr1)**2/vnr1**2*self.inner(x, v1, v)/(1+jnp.cos(t*vnr1))*v1

    def gamma(self, x, xi, eta):
        """Christoffel function
        """
        return 1/self.r**2*x*jnp.sum(xi*eta)

    def ito_drift(self, x):
        return -1*self.dim/(2*self.r**2) * x

    def retract(self, x, v):
        return (x+v)/jnp.sqrt(jnp.sum((x+v)**2))*self.r

    def approx_nearest(self, q):
        """ find point on the manifold that
        is nearest to q, same order as the nearest point
        """
        return q/jnp.sqrt(jnp.sum(q*q))*self.r

    def sigma(self, x, dw):
        return dw

    @partial(jax.jit, static_argnums=(0,))
    def make_tangent_basis(self, y):
        """ yet another way to get complement basis
        """
        cmp = complement_basis_for_vector(y)
        mat = jax.vmap(lambda v1: jnp.array(
            [self.inner(y, v1, cmp[:, i]) for i in range(cmp.shape[1])]),
                       in_axes=1,
                       )(cmp)

        ei, ev = jla.eigh(mat)
        return cmp@ev@(1/jnp.sqrt(ei)[:, None]*ev.T)
