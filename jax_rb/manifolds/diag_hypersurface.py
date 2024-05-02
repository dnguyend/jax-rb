"""Hypersurface with a constraint of the form :math:`\\sum_i d_i x_i^p = 1`
"""
import jax.numpy as jnp
import jax.numpy.linalg as jla
from .global_manifold import GlobalManifold
from ..utils.utils import (grand)


class DiagHypersurface(GlobalManifold):
    """Hypersurface of the form :math:`\\sum_i d_ix_i^p = 1`.
    
    :param dvec: vector :math:`d_i` of coefficients.
       Sort dvec so dvec[-1] is positive.
    :param p: :math:`p > 0` is an integer, degree of the constraint.

    Use embedded metric.
    """
    def __init__(self, dvec, p):
        self.dvec = dvec
        self.shape = dvec.shape
        self.p = p
        self.dim = dvec.shape[0]-1

    def name(self):
        return f"DH{self.shape[0]-1}, {self.p}"

    def g_metric(self, x, omg):
        return omg

    def inv_g_metric(self, x, omg):
        return omg

    def inner(self, x, a, b):
        return jnp.sum(a*b)

    def cfunc(self, x):
        """ constraint for the surface is cfunc(x) = 1
        """
        return jnp.sum(self.dvec*x**self.p)

    def grad_c(self, x):
        """ gradient of cfunc
        """
        return self.p*self.dvec*x**(self.p-1)

    def rand_point(self, key):
        """random point on  manifold
        """
        p = self.p
        dvec = self.dvec
        x, key = grand(key, self.shape)
        val = self.cfunc(x)
        if p % 2 == 1:
            return x/jnp.abs(val)**(1/p)*jnp.sign(val)
        if val < 0:
            ret = jnp.concatenate(
                [x[:-1],
                 jnp.array([1/dvec[-1]*(1-jnp.sum(dvec[:-1]*x[:-1]**p))**(1/p)])])
        else:
            ret = x/val**(1/p)
        return ret, key

    def rand_vec(self, key, x):
        """random tangent vector
        """
        omg, key = grand(key, self.shape)
        return self.proj_scale(x, omg), key

    def proj(self, x, omg):
        """ othogonal projection
        """
        gcx = self.grad_c(x)
        return omg - gcx*jnp.sum(gcx*omg)/jnp.sum(gcx*gcx)

    def approx_nearest(self, q):
        """ tubular retraction. Need some work
        to show this is actually approx_nearest
        """
        val = self.cfunc(q)
        return q/val**(1/self.p)

    def retract(self, x, v):
        return self.approx_nearest(x + v - 0.5*self.proj_scale(x, self.gamma(x, v, v)))

    def proj_scale(self, x, omg):
        """rescale projection
        """
        return omg - x*jnp.sum(self.dvec*x**(self.p-1)*omg)

    def gamma(self, x, xi, eta):
        """Christoffel function
        """
        p = self.p
        gcx = self.grad_c(x)
        return p*(p-1)*gcx*jnp.sum(self.dvec*x**(p-2)*xi*eta)/jnp.sum(gcx*gcx)

    def ito_drift(self, x):
        p = self.p
        gcx = self.grad_c(x)
        return -0.5*p*(p-1)*gcx*(
            jnp.sum(self.dvec*x**(p-2)) - jnp.sum(self.dvec*x**(p-2)*gcx*gcx)/jnp.sum(gcx*gcx)
        )/jnp.sum(gcx*gcx)

    def pseudo_transport(self, x, y, v):
        gcx = self.grad_c(x)
        gcy = self.grad_c(y)

        a = jnp.sum(gcy*v)*(jla.norm(gcx)*jla.norm(gcy) - jnp.sum(gcx*gcy)) \
             / (jnp.sum(gcx*gcx)*jnp.sum(gcy*gcy) - jnp.sum(gcy*gcx)**2)
        return v - a*gcx - (jnp.sum(gcy*v) - a*jnp.sum(gcy*gcx))/jnp.sum(gcy*gcy)*gcy

    def sigma(self, x, dw):
        return dw

    def rtr_tan_scale(self, yv, dyv):
        """retraction to the tangent bundle using rescale
        projection
        """
        y1 = self.retract(yv[:, 0], dyv[:, 0])
        v1 = self.proj_scale(y1, yv[:, 1] + dyv[:, 1])
        v1 = v1*jnp.sqrt(self.inner(yv[0], yv[:, 1], yv[:, 1])/self.inner(y1, v1, v1))
        return jnp.concatenate([
            y1[:, None],
            v1[:, None]], axis=1)

    def geodesic(self, x, v, t, nstep=100):
        """ approximate geodesic
        using the retraction to the tangent bundle rtr_tan
        """
        yv = jnp.concatenate([x[:, None], v[:, None]], axis=1)
        h = t/nstep

        def dyvdt(_, yv):
            return jnp.concatenate(
                [yv[:, [1]], -self.gamma(yv[:, 0], yv[:, 1], yv[:, 1])[:, None]],
                axis=1)

        t0 = 0
        for _ in range(1, nstep+1):
            # Apply Runge Kutta Formulas to find next value of y
            k1 = h * dyvdt(t0, yv)
            k2 = h * dyvdt(t0 + 0.5 * h, yv + 0.5 * k1)
            k3 = h * dyvdt(t0 + 0.5 * h, yv + 0.5 * k2)
            k4 = h * dyvdt(t0 + h, yv + k3)

            yv = self.rtr_tan_scale(yv,  (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4))
            t0 = t0 + h
        return yv[:, 0], yv[:, 1]

    def make_tangent_basis(self, x):
        """ tangent  basis at x
        """
        d = self.dim
        gcx = self.grad_c(x)
        proj_mat = jnp.eye(self.shape[0]) - gcx[:, None]@gcx[None, :]/jnp.sum(gcx*gcx)
        _, ev = jla.eigh(proj_mat)
        cmat = ev[:, 1:]

        mat = jnp.empty((d, d))
        for i in range(d):
            for j in range(d):
                mat = mat.at[i, j].set(self.inner(x, cmat[:, i], cmat[:, j]))
        ei, ev = jla.eigh(mat)
        return cmat@ev@(1/jnp.sqrt(jnp.abs(ei))[:, None]*ev.T)
