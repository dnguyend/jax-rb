"""test basic functionalities of diag_hypersurface
"""
import jax
import jax.numpy as jnp
from jax import random, jvp, grad, jit

from jax_rb.manifolds.diag_hypersurface import DiagHypersurface
from jax_rb.utils.utils import (grand, tv_mode_product)


def test_diag_hypersurface():
    n = 6
    p = 4
    key = random.PRNGKey(0)
    dvec, key = grand(key, (n,))
    dvec = dvec.at[-1].set(1.)

    mnf = DiagHypersurface(dvec, p)
    x, key = mnf.rand_point(key)
    print(grad(mnf.cfunc)(x))
    print(mnf.grad_c(x))
    while True:
        q, key = mnf.rand_ambient(key)
        if mnf.cfunc(q) > 0:
            xq = mnf.approx_nearest(q)
            break
    print(mnf.cfunc(q), mnf.cfunc(xq))

    # now tangent.
    omg, key = mnf.rand_ambient(key)
    p1 = mnf.proj_scale(x, omg)
    print(jnp.sum(mnf.grad_c(x)*p1))

    p2 = mnf.proj(x, omg)
    print(jnp.sum(mnf.grad_c(x)*p2))

    xi, key = mnf.rand_vec(key, x)
    eta, key = mnf.rand_vec(key, x)
    print(jvp(lambda x: mnf.inner(x, eta, eta), (x,), (xi,))[1])
    print(2*mnf.inner(x, eta, mnf.gamma(x, xi, eta)))
    print(jvp(lambda x: mnf.proj(x, eta), (x,), (xi,))[1] + mnf.gamma(x,  xi,  eta))

    def check_ito_drift(self, x):
        s1 = self.ito_drift(x)
        p = self.shape[0]
        s = jnp.zeros(p)
        for i in range(p):
            ei = jnp.zeros(self.shape).at[i].set(1.)
            s -= self.gamma(x, ei, self.proj(x, self.inv_g_metric(x, ei)))
        print(2*s1, s)

    check_ito_drift(mnf, x)

    y, key = mnf.rand_point(key)
    v, key = mnf.rand_vec(key, x)

    w = mnf.pseudo_transport(x, y, v)
    print(mnf.inner(x, v, v))
    print(mnf.inner(y, w, w))

    dlt = 1e-5
    # need higher precision here.
    y1, v1 = mnf.geodesic(x, xi, 1, 100)
    _, v1a = mnf.geodesic(x, xi, 1+dlt, 100)

    print((v1a-v1)/dlt + mnf.gamma(y1, v1, v1))


    print(mnf.grad_c(x)[None, :]@mnf.make_tangent_basis(x))

    f1, key = grand(key, (n,))
    f2, key = grand(key, (n, n))
    f3, key = grand(key, (n, n, n))

    @jit
    def f(x):
        return jnp.sum(f1*x) + jnp.sum(x*(f2@x)) + tv_mode_product(f3, x, 3)

    egradf = jit(grad(f))

    @jit
    def ehessf(x, omg):
        return jvp(egradf, (x,), (omg,))[1]

    def lb_test(self, x, f):
        n = self.shape[0]
        ret = 0
        rgradf = jit(lambda x: self.proj(
            x,
            self.inv_g_metric(x, grad(f)(x))))
        obs = self.make_tangent_basis(x)
        for i in range(n-1):
            tmp = jvp(rgradf, (x,), (obs[:, i],))
            nxi = tmp[1] + self.gamma(x, obs[:, i], tmp[0])
            ret += self.inner(x, obs[:, i], nxi)
        return ret

    print(lb_test(mnf, x, f))
    print(mnf.laplace_beltrami(x, egradf(x), ehessf))


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    test_diag_hypersurface()
