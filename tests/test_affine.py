"""test basic functionalities of symmetric positive definite
"""
import jax
import jax.numpy as jnp
from jax import random, jvp, grad

from jax_rb.manifolds.affine_left_invariant import AffineLeftInvariant
from jax_rb.utils.utils import (grand, rand_positive_definite)

jax.config.update("jax_enable_x64", True)


def test_affine():
    key = random.PRNGKey(0)
    jax.config.update("jax_enable_x64", True)

    n = 3

    metric_mat, key = rand_positive_definite(key, (n*(n+1)))
    mnf = AffineLeftInvariant(n, metric_mat)

    x, key = mnf.rand_point(key)

    # check metric compatibility
    va, key = mnf.rand_vec(key, x)
    vb, key = mnf.rand_vec(key, x)
    omg, key = mnf.rand_ambient(key)
    omga, key = mnf.rand_ambient(key)

    print(mnf.inner(x, omg, omga))
    print(mnf.inner(x, omga, omg))

    omg1 = mnf.g_metric(x, omg)
    omg2 = mnf.inv_g_metric(x, omg1)
    print(omg2 - omg)
    print(jnp.sum(va*omg1) - mnf.inner(x, va, omg))

    pomg = mnf.proj(x, omg)
    # print(jnp.sum(x*pomg))
    print(mnf.inner(x, omg, va) - mnf.inner(x, mnf.proj(x, omg), va))

    print(mnf.inner(x,
                    mnf.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[1, 2].set(1)),
                    mnf.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[1, 2].set(1))))

    print(mnf.inner(x,
                    mnf.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[0, 2].set(1)),
                    mnf.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[1, 2].set(1))))

    print(jvp(lambda x: mnf.inner(x, vb, vb), (x,), (va,))[1])
    print(2*mnf.inner(x, vb,
                      jvp(lambda x: mnf.proj(x, vb), (x,), (va,))[1]
                      + mnf.gamma(x, va, vb)))

    # now check the drift
    def check_ito_drift(self, x):
        s1 = self.ito_drift(x)
        n, p = self.shape
        s = jnp.zeros((n, p))
        for i in range(n):
            for j in range(p):
                eij = jnp.zeros(self.shape).at[i, j].set(1.)
                s -= self.gamma(x, eij, self.proj(x, self.inv_g_metric(x, eij)))
        print(2*s1 - s)

    check_ito_drift(mnf, x)

    def check_v0(self):
        # compute as derivative of projection
        p, i_sqrt_g_mat = self.shape[0], self._i_sqrt_g_mat
        vv = jnp.zeros((p, p))
        zr = jnp.zeros((p, p))
        for i in range(p-1):
            for j in range(p):
                vij = self._mat_apply(i_sqrt_g_mat,
                                      zr.at[i, j].set(1.))

                vv += self.gamma(jnp.eye(p), vij, vij)
                vv += jvp(lambda x: mnf.sigma(x, zr.at[i, j].set(1.)),
                          (jnp.eye(p),), (vij,))[1]
                
        return -0.5*vv

    print(check_v0(mnf) - mnf.v0)
    
    # now test Laplacian

    f1, key = grand(key, (n+1, n+1))
    f2, key = grand(key, ((n+1)**2, (n+1)**2))
    f3, key = grand(key, ((n+1)**2, (n+1)**2))

    @jax.jit
    def f(U):
        return jnp.sum(f1*U) + jnp.sum(U.reshape(-1)*(f2@U.reshape(-1))) \
            + jnp.sum(U.reshape(-1)*(f2@U.reshape(-1)))**2

    egradf = jax.jit(jax.grad(f))

    @jax.jit
    def ehessf(U, omg):
        return jvp(egradf, (U,), (omg,))[1]

    def lp_test(self, x, f):
        n = self.shape[0]
        ret = 0
        rgradf = jax.jit(lambda x: self.proj(
            x,
            self.inv_g_metric(x, grad(f)(x))))
        for i in range(n-1):
            for j in range(n):
                vij = self.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[i, j].set(1.))
                tmp = jvp(rgradf, (x,), (vij,))
                nxi = tmp[1] + self.gamma(x, vij, tmp[0])
                ret += self.inner(x, vij, nxi)
        return ret

    print(lp_test(mnf, x, f))
    print(mnf.laplace_beltrami(x, egradf(x), ehessf))


if __name__ == '__main__':
    test_affine()
    
