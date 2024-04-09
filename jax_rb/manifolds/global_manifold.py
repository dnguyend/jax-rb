"""Base class for manifold in global embedded coordinates
"""
from functools import partial
from abc import ABCMeta, abstractmethod

import jax
import jax.numpy as jnp
from ..utils.utils import (grand)


class GlobalManifold(metaclass=ABCMeta):
    """A manifold :math:`\\mathcal{M}` embedded in a vector space :math:`\\mathcal{E}` .
    """
    @abstractmethod
    def __init__(self):
        """Constructor

        :param shape: shape of the ambient vector space,
        :param dim: dimension of the manifold.
        """
        self.shape = None
        self.dim = None
        raise NotImplementedError

    def name(self):
        """ name of the manifold.
        """
        raise NotImplementedError

    def inner(self, x, a, b):
        """ Riemannian inner product.

        :param a: a vector in ambient space,
        :param b: a vector in ambient space,
        :return: the inner product of a and b using the metric :math:`\\mathsf{g}` .
        """
        raise NotImplementedError

    def g_metric(self, x, omg):
        """ the metric operator g, which is symmetric. The corresponding metric is
        :math:`\\langle \\omega, g(x)\\omega \\rangle_{\\mathcal{E}}` .
        """
        raise NotImplementedError

    def inv_g_metric(self, x, omg):
        """ inverse of the metric operator g.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def proj(self, x, omg):
        """ Metric compatible projection
        :param x: a point on the manifold,
        :param omg: a vector on the ambient manifold :math:`\\mathcal{E}` ,
        :returns: a point the tangent space at x.

        """
        raise NotImplementedError

    def rand_ambient(self, key):
        """Random ambient vector.
        """
        return grand(key, self.shape)

    def rand_vec(self, key, x):
        """Random tangent vector at x.
        """
        omg, key = grand(key, self.shape)
        return self.proj(x, omg), key

    def rand_point(self, key):
        """ A random point on the manifold.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def gamma(self, x, xi, eta):
        """Christoffel function. Symmetric for two tangent vectors xi, eta.
        The corresponding Levi-Civita connection is
        :math:`\\nabla_{\\mathtt{X}}\\mathtt{Y} = \\mathrm{D}_{\\mathtt{X}}\\mathtt{Y} + \\Gamma(x; \\mathtt{X}, \\mathtt{Y})`
        for two vector fields :math:`\\mathtt{X}, \\mathtt{Y}`.
        """
        raise NotImplementedError

    def retract(self, x, v):
        """ Second order retraction

        :param x: a point on the manifold,
        :param v: a tangent vector at x,
        :returns: a point on the manifold.
        """
        # x1 = x + v - 0.5* self.proj(x, self.gamma(x, v, v))
        # return jax.scipy.linalg.polar(x1)[0]
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def ito_drift(self, x):
        """Ito Brownian drift as an ambient vector.
        """
        raise NotImplementedError

    # @partial(jax.jit, static_argnums=(0,))
    def laplace_beltrami(self, x, egradx, ehessvp):
        """ Laplace Beltrami operator. This works in for vector and matrices. For a specific manifold, this may be simplified. 
        We assume f is a scalar function in a tubular neighborhood of the manifold.

        :param x: a point on the manifold,
        :param egradx: is the Euclidean gradient of :math:`f` , a matrix of the same shape with x,
        :param ehessvp: is the Euclidean Hessian Productof :math:`f` , a linear operator on :math:`\\mathcal{E}` ,
        :returns: the value of the Laplace Beltrami operator of :math:`f` .
        """
        ret = 0
        ndim = jnp.prod(jnp.array(self.shape))
        for i in range(ndim):
            e_i = jnp.zeros(ndim).at[i].set(1.).reshape(self.shape)
            ret += self.proj(x, self.inv_g_metric(
                x, ehessvp(x, e_i))).reshape(-1)[i]
        return ret + 2*jnp.sum(self.ito_drift(x)*egradx)

    def pseudo_transport(self, x, y, v):
        """ an approximate parallel transport from x to y

        :param x: a point on the manifold,
        :param y: a point on the manifold,
        :param v: a tangent vector at x,

        :returns: a tangent vector at y.
        """
        raise NotImplementedError

    def sigma(self, x, dw):
        """ Sigma map to generate Brownian motion.

        :param x: a point on the manifold,
        :param dw: a point on the ambient space,
        :return: apoint on the ambient space such that :math:`\\Pi(x) \\sigma(x)  \\sigma^{\\mathsf{T}}(x)\\mathsf{g}^{-1}(x) dw = \\Pi(x)dw`
        """
        raise NotImplementedError
