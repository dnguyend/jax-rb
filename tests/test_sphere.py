"""test basic functionalities of sphere
"""
import jax
import jax.numpy as jnp
from jax import random, jvp, grad

import jax_rb.manifolds.sphere as spm
from jax_rb.utils.utils import (grand, tv_mode_product)

jax.config.update("jax_enable_x64", True)


def test_sphere():
    """test
    """
    jax.config.update('jax_default_device', jax.devices('cpu')[0])

    n = 4
    r = 1.3
    sph = spm.Sphere(n, r)
    key = random.PRNGKey(0)
    x, key = sph.rand_point(key)
    print(jnp.sum(x*x) - r**2)
    y, key = sph.rand_point(key)

    v, key = sph.rand_vec(key, x)
    print(sph.inner(x, v, x))

    # check metric compatibility
    va, key = sph.rand_vec(key, x)
    vb, key = sph.rand_vec(key, x)
    omg, key = sph.rand_ambient(key)

    print(sph.inner(x, omg, va) - sph.inner(x, sph.proj(x, omg), va))

    print(jvp(lambda x: sph.inner(x, vb, vb), (x,), (va,))[1])
    print(jvp(lambda x: sph.proj(x, vb), (x,), (va,))[1]
          +  sph.gamma(x, va, vb))

    # check geodesic
    t = 3.
    xt = sph.exp(x, t*va)
    print(sph.inner(x, xt, xt) - r**2)

    dvt = jvp(lambda t: sph.exp(x, t*va), (t,), (1.,))[1]
    print(sph.d_exp(x, va, t) - dvt)
    ddvt = jvp(lambda t: sph.d_exp(x, va, t), (t,), (1.,))[1]
    print(sph._d2_exp(x, va, t) - ddvt)

    # test geodesic
    print(ddvt + sph.gamma(xt, dvt, dvt))

    # test parallel transport
    vap = sph.pseudo_transport(x, y, va)
    print(sph.inner(y, y, vap))
    print(sph.inner(y, vap, vap) - sph.inner(x, va, va))

    v1 = sph.log(x, y)
    print(sph.inner(x, x, v1))
    print(y - sph.exp(x, v1))

    t1 = 1.

    print(vap - sph.transport_along_geodesic(x, v1, va, t1))
    dva1 = jvp(lambda t: sph.transport_along_geodesic(x, v1, va, t), (t1,), (1.,))[1]
    dv1 = sph.d_exp(x, v1, t1)
    print(dva1 + sph.gamma(y, dv1, vap))

    print(sph.exp(x, v1) - y)
    print(jnp.sqrt(sph.inner(x, v1, v1)) - sph.dist(x, y))

    print(x[None, :]@sph.make_tangent_basis(x))

    # now check the sum
    def check_ito_drift(self, x):
        s1 = self.ito_drift(x)
        p = self.shape[0]
        s = jnp.zeros(p)
        for i in range(p):
            ei = jnp.zeros(self.shape).at[i].set(1.)
            s -= self.gamma(x, ei, self.proj(x, self.inv_g_metric(x, ei)))
        print(2*s1, s)

    check_ito_drift(sph, x)

    f1, key = grand(key, (n,))
    f2, key = grand(key, (n, n))
    f3, key = grand(key, (n, n, n))

    @jax.jit
    def f(x):
        return jnp.sum(f1*x) + jnp.sum(x*(f2@x)) + tv_mode_product(f3, x, 3)

    egradf = jax.jit(jax.grad(f))

    @jax.jit
    def ehessf(x, omg):
        return jvp(egradf, (x,), (omg,))[1]

    def lb_test(self, x, f):
        n = self.shape[0]
        ret = 0
        rgradf = jax.jit(lambda x: self.proj(
            x,
            self.inv_g_metric(x, grad(f)(x))))
        obs = self.make_tangent_basis(x)
        for i in range(n-1):
            tmp = jvp(rgradf, (x,), (obs[:, i],))
            nxi = tmp[1] + self.gamma(x, obs[:, i], tmp[0])
            ret += self.inner(x, obs[:, i], nxi)
        return ret

    print(lb_test(sph, x, f))
    print(sph.laplace_beltrami(x, egradf(x), ehessf))


if __name__ == '__main__':
    test_sphere()
