import jax
import jax.numpy as jnp
from jax import random, jvp, grad

from jax_rb.manifolds.se_left_invariant import SELeftInvariant
from jax_rb.utils.utils import (grand, sym, rand_positive_definite)

jax.config.update("jax_enable_x64", True)


def test_se():
    key = random.PRNGKey(0)
    jax.config.update("jax_enable_x64", True)

    n = 4

    metric_mat, key = rand_positive_definite(key, (n*(n+1))//2)
    mnf = SELeftInvariant(n, metric_mat)

    x, key = mnf.rand_point(key)
    print(x[:-1, :-1]@x[:-1, :-1].T - jnp.eye(n))

    v, key = mnf.rand_vec(key, x)
    print(sym(x[:-1, :-1].T@v[:-1, :-1]))

    # check metric compatibility
    va, key = mnf.rand_vec(key, x)
    vb, key = mnf.rand_vec(key, x)
    omg, key = mnf.rand_ambient(key)

    omg1 = mnf.g_metric(x, omg)
    omg2 = mnf.inv_g_metric(x, omg1)
    print(omg2 - omg)
    print(jnp.sum(va*omg1) - mnf.inner(x, va, omg))

    # print(jnp.sum(x*pomg))
    print(mnf.inner(x, omg, va) - mnf.inner(x, mnf.proj(x, omg), va))

    print(jvp(lambda x: mnf.inner(x, vb, vb), (x,), (va,))[1])
    print(2*mnf.inner(x, vb, 
                      jvp(lambda x: mnf.proj(x, vb), (x,), (va,))[1]
                      + mnf.gamma(x, va, vb)))

    D1 = jvp(lambda x: mnf.proj(x, vb), (x,), (va,))[1] + mnf.gamma(x, va, vb)
    print(sym(x[:-1, :-1].T@D1[:-1, :-1]))

    def check_v0(self):
        p, i_sqrt_mat = self.shape[0], self._i_sqrt_g_mat
        vv = jnp.zeros((p, p))
        zr = jnp.zeros((p, p))
        for i in range(1, p-1):
            for j in range(i):
                eij = zr.at[i, j].set(1.).at[j, i].set(-1.)
                eij = 1/jnp.sqrt(2)*self._mat_apply(i_sqrt_mat, eij)
                vv += eij@eij + self.gamma(jnp.eye(p), eij, eij)
        for i in range(p-1):
            eij = self._mat_apply(i_sqrt_mat, zr.at[i, -1].set(1.))
            vv += eij@eij + self.gamma(jnp.eye(p), eij, eij)

        return -0.5*vv

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
    
    check_ito_drift(mnf, x)    
    print(check_v0(mnf))
    print(mnf.v0)
    

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

    def lb_test(self, x, f):
        n1 = self.shape[0]
        isqrt2 = 1/jnp.sqrt(2)
        ret = 0
        rgradf = jax.jit(lambda x: self.proj(
            x,
            self.inv_g_metric(x, grad(f)(x))))
        zr = jnp.zeros((n1, n1))
        for i in range(1, n1-1):
            for j in range(i):
                vij = self.left_invariant_vector_field(x,
                                                       isqrt2*zr.at[i, j].set(1.).at[j, i].set(-1))
                tmp = jvp(rgradf, (x,), (vij,))
                nxi = tmp[1] + self.gamma(x, vij, tmp[0])
                ret += self.inner(x, vij, nxi)
        for i in range(n1-1):
            vij = self.left_invariant_vector_field(x, zr.at[i, -1].set(1.))
            tmp = jvp(rgradf, (x,), (vij,))
            nxi = tmp[1] + self.gamma(x, vij, tmp[0])
            ret += self.inner(x, vij, nxi)

        return ret

    print(lb_test(mnf, x, f))
    print(mnf.laplace_beltrami(x, egradf(x), ehessf))

    
if __name__ == '__main__':
    test_se()
    
