"""Base class for matrix groups with left invariant metrics.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from .global_manifold import GlobalManifold
from ..utils.utils import (grand, lie)


class MatrixLeftInvariant(GlobalManifold):
    """Matrix group with left invariant metric.
    
    :param p: the size of the matrix
    :param g_mat: The matrix defining the inner product at the identity. Typically g_mat is of size :math:`\\dim \\mathrm{G}` .
    """
    def __init__(self, p, g_mat):
        """Constructor
        """
        self.shape = (p, p)
        self.dim = p*p
        self._g_mat = g_mat
        ei, ev = jla.eigh(g_mat)
        self._i_sqrt_g_mat = ev@((1/jnp.sqrt(ei))[:, None]*ev.T)
        # Stratonovich drift at id
        self.v0 = self._make_v0() #: Stratonovich drift at the identity.
        self.id_drift = self._make_id_drift()  #: Ito drift at the identity.

    def name(self):
        raise NotImplementedError

    def _lie_algebra_proj(self, omg):
        """ The projection :math:`p_{\\mathfrak{g}}` at the identity.
        """
        raise NotImplementedError

    def _mat_apply(self, mat, omg):
        """ Implementing the operator :math:`\\mathcal{I}` applied on omg in :math:`\\mathcal{E}`.
        """
        raise NotImplementedError

    def _id_opt(self, omg):
        """The metric applied at the identity.
        """
        return self._mat_apply(self._g_mat, omg)

    def _inv_id_opt(self, omg):
        """Invert of _id_opt.
        """
        return self._mat_apply(jla.inv(self._g_mat), omg)

    def proj(self, x, omg):
        return x@self._lie_algebra_proj(jla.solve(x, omg))

    def _d_proj(self, x, xi, eta):
        ivx = jla.inv(x)
        return xi@self._lie_algebra_proj(jla.solve(x, eta)) \
            - x@self._lie_algebra_proj(ivx@xi@ivx@eta)

    def rand_ambient(self, key):
        return grand(key, (self.shape))

    def rand_vec(self, key, x):
        omg, key = grand(key, self.shape)
        return self.proj(x, omg), key

    def rand_point(self, key):
        raise NotImplementedError

    def inner(self, x, a, b):
        return jnp.sum(jla.solve(x, a)*self._id_opt(jla.solve(x, b)))

    def g_metric(self, x, omg):
        return jla.solve(x.T, self._id_opt(jla.solve(x, omg)))

    def inv_g_metric(self, x, omg):
        return x@self._inv_id_opt(x.T@omg)

    @partial(jax.jit, static_argnums=(0,))
    def gamma(self, x, xi, eta):
        # return - self.d_proj(x, xi, eta)
        #    + self.proj(x, self.gamma_ambient(x, xi, eta))
        ivxi = jla.solve(x, xi)
        iveta = jla.solve(x, eta)

        return -0.5*(xi@iveta + eta@ivxi) \
            + 0.5*x@self._inv_id_opt(
                self._lie_algebra_proj(
                    lie(self._id_opt(ivxi), iveta.T) \
                    + lie(self._id_opt(iveta), ivxi.T)))

    @partial(jax.jit, static_argnums=(0,))
    def gamma_ambient(self, x, xi, eta):
        """Christoffel function for ambient manifold.
        """
        ivx = jla.inv(x)
        return 0.5*x@self._inv_id_opt(
            - self._id_opt(ivx@xi@ivx@eta + ivx@eta@ivx@xi) \
            + lie(self._id_opt(ivx@xi), eta.T@ivx.T) \
            + lie(self._id_opt(ivx@eta), xi.T@ivx.T))

    def retract(self, x, v):
        raise NotImplementedError

    def _make_id_drift_longer(self):
        """make the drift at identity.
        The longer way, sum gamma.x.
        """
        p = self.shape[0]
        drft = jnp.zeros(self.shape)
        for i in range(p):
            for j in range(p):
                eij = jnp.zeros((p, p)).at[i, j].set(1.)
                drft -= self.gamma(jnp.eye(p), eij,
                                   self._lie_algebra_proj(self._inv_id_opt(eij)))
        return 0.5*drft

    def _make_id_drift(self):
        """make the drift at identity.
        Simplify so we dont need to evaluate gamma.
        """
        p = self.shape[0]
        v = jnp.zeros((p, p))
        zr = jnp.zeros((p, p))

        def lp(a):
            return self._lie_algebra_proj(a)
        for i in range(p):
            for j in range(p):
                eij = zr.at[i, j].set(1.)
                v += - eij@self._inv_id_opt(lp(eij)) \
                    + self._inv_id_opt(lp(lie(lp(eij), eij.T)))

        return -0.5*v


    def _make_v0(self):
        """ make v0, the identity tangent vector corresponding to
        the Stratonovich drift.
        """
        p = self.shape[0]
        v = jnp.zeros((p, p))
        zr = jnp.zeros((p, p))

        for i in range(p):
            for j in range(p):
                eij = zr.at[i, j].set(1.)

                v += self._inv_id_opt(
                    self._lie_algebra_proj(
                        lie(self._lie_algebra_proj(eij), eij.T)))

        return -0.5*v

    def approx_nearest(self, q):
        """ find point on the manifold that
        is nearest to q, same order as the nearest point.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def ito_drift(self, x):
        return x@self.id_drift


    @partial(jax.jit, static_argnums=(0,))
    def stratonovich_drift(self, x):
        """ Stratonovich drift.
        """
        return x@self.v0

    # @partial(jax.jit, static_argnums=(0,))
    def laplace_beltrami(self, x, egradx, ehessvp):
        p = self.shape[0]
        tup = jnp.zeros((p, p))
        ret = 0
        for i in range(p):
            for j in range(p):
                e_ij = tup.at[i, j].set(1.)
                ret += self.proj(x, self.inv_g_metric(
                    x, ehessvp(x, e_ij)))[i, j]
        return ret + 2*jnp.sum(self.ito_drift(x)*egradx)

    def left_invariant_vector_field(self, x, v):
        """ map from a unit vector in the trace metric
        to a vector field with unit length in the
        left invariant metric.
        """
        return x@self._mat_apply(self._i_sqrt_g_mat, v)

    @partial(jax.jit, static_argnums=(0,))
    def pseudo_transport(self, x, y, v):
        """the easy one
        """
        return y@jla.solve(x, v)

    @partial(jax.jit, static_argnums=(0,))
    def sigma_id(self, dw):
        """ sigma, to generate the Brownian motion at the identity.
        """
        return self._lie_algebra_proj(self._mat_apply(self._i_sqrt_g_mat, dw))

    @partial(jax.jit, static_argnums=(0,))
    def sigma(self, x, dw):
        """ sigma, to generate the Brownian motion.
        """
        return x@self.sigma_id(dw)
