"""test basic functionalities of symmetric positive definite
"""
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random, jvp, grad

from jax_rb.manifolds.spd import PositiveDefiniteManifold
from jax_rb.utils.utils import (grand, sym, asym)

jax.config.update("jax_enable_x64", True)


def test_spd():
    key = random.PRNGKey(0)

    n = 5

    spd = PositiveDefiniteManifold(n)
    x, key = spd.rand_point(key)
    print(x - x.T)
    print(jla.eigh(x)[0])
    y, key = spd.rand_point(key)

    v, key = spd.rand_vec(key, x)
    print(v - v.T)

    # check metric compatibility
    va, key = spd.rand_vec(key, x)
    vb, key = spd.rand_vec(key, x)
    omg, key = spd.rand_ambient(key)

    omg1 = spd.g_metric(x, omg)
    omg2 = spd.inv_g_metric(x, omg1)
    print(omg2 - omg)
    print(jnp.sum(va*omg1) - spd.inner(x, va, omg))

    pomg = spd.proj(x, omg)
    print(asym(pomg))
    print(spd.inner(x, omg, va)
          - spd.inner(x, spd.proj(x, omg), va))

    print(spd.gamma(x, va, vb))

    print(jvp(lambda x: spd.inner(x, vb, vb), (x,), (va,))[1])
    print(2*spd.inner(x, vb,
                     jvp(lambda x: spd.proj(x, vb), (x,), (va,))[1] + spd.gamma(x, va, vb)))

    # now check the sum
    def check_ito_drift(self, x):
        s1 = self.ito_drift(x)
        p = self.shape[0]
        s = jnp.zeros((p, p))
        for i in range(p):
            for j in range(p):
                eij = jnp.zeros(self.shape).at[i, j].set(1.)
                # s -= self.gamma(x, eij, self.proj(x, self.inv_g_metric(x, eij)))
                # s += sym(eij@jla.inv(x)@sym(x@eij@x))
                # s += sym(eij@(eij@x+eij.T@x))
                s -= self.gamma(x, eij, self.proj(x, self.inv_g_metric(x, eij)))
        print(2*s1, s, 2*s1 - s)

    check_ito_drift(spd, x)

    @jax.jit
    def xhf(x):
        ei, ev = jla.eigh(x)
        return ev@(jnp.sqrt(ei)[:, None]*ev.T)

    # now compute nabla
    def check_two(self, x):
        p = self.shape[0]
        s = jnp.zeros((p, p))
        s1 = jnp.zeros((p, p))

        def vfield(x, omg):
            x2 = xhf(x)
            return x2@omg@x2

        for i in range(p):
            for j in range(p):
                eij = jnp.zeros(self.shape).at[i, j].set(1.)
                seij = sym(eij)
                v1 = vfield(x, sym(eij))
                dxh = jax.jvp(xhf, (x,), (v1,))
                s1 -= jax.jvp(lambda x: vfield(x, sym(eij)), (x,), (v1,))[1]
                s -= 2*sym(dxh[0]@seij@dxh[1])

        return s, s1
    ret = check_two(spd, x)
    print(0.5*ret[0]+spd.ito_drift(x))
    # print(0*5*ret[0]+spd.ito_drift(x))
    print(spd.stratonovich_drift(x))

    f1, key = grand(key, (n, n))
    f2, key = grand(key, (n, n))
    f2a, key = grand(key, (n, n))

    def f(q):
        return jnp.sum(f1*(q@q.T)) + jnp.trace(q@q.T@f2@q@q.T@f2a)

    egradf = jax.jit(jax.grad(f))
    
    def jvpf(q, omg):
        return jnp.sum(f1*(omg@q.T)) +jnp.sum(f1*(q@omg.T))\
            + jnp.trace(omg@q.T@f2@q@q.T@f2a) + jnp.trace(q@omg.T@f2@q@q.T@f2a) + jnp.trace(q@q.T@f2@omg@q.T@f2a) + jnp.trace(q@q.T@f2@q@omg.T@f2a)

    def hessf(q, omg1, omg2):
        return jnp.sum(f1*(omg1@omg2.T)) +jnp.sum(f1*(omg2@omg1.T))\
            + jnp.trace(omg1@omg2.T@f2@q@q.T@f2a) + jnp.trace(omg1@q.T@f2@omg2@q.T@f2a) + jnp.trace(omg1@q.T@f2@q@omg2.T@f2a) \
            + jnp.trace(omg2@omg1.T@f2@q@q.T@f2a) + jnp.trace(q@omg1.T@f2@omg2@q.T@f2a) + jnp.trace(q@omg1.T@f2@q@omg2.T@f2a) \
            + jnp.trace(omg2@q.T@f2@omg1@q.T@f2a) + jnp.trace(q@omg2.T@f2@omg1@q.T@f2a) + jnp.trace(q@q.T@f2@omg1@omg2.T@f2a) \
            + jnp.trace(omg2@q.T@f2@q@omg1.T@f2a) + jnp.trace(q@omg2.T@f2@q@omg1.T@f2a) + jnp.trace(q@q.T@f2@omg2@omg1.T@f2a)

    @jax.jit
    def ehessf(x, omg):
        return jvp(egradf, (x,), (omg,))[1]

    def make_proj_mat(self, x):
        vshape = jnp.prod(jnp.array(self.shape))
        mat = jnp.empty((vshape,  vshape))
        for i in range(vshape):
            mat = mat.at[:, i].set(
                self.proj(x, jnp.zeros(vshape).at[i].set(1).reshape(self.shape)).reshape(-1))
        return mat
    pmat = make_proj_mat(spd, x)
    print((pmat@omg.reshape(-1)).reshape(spd.shape) - spd.proj(x, omg))

    def make_tangent_basis(self, x):
        d = self.dim

        pmat = make_proj_mat(self, x)
        cmat = jla.eigh(pmat)[1][:, -d:]
        mat = jnp.empty((d, d))
        for i in range(d):
            for j in range(d):
                mat = mat.at[i, j].set(self.inner(x, cmat[:, i].reshape(self.shape),
                                                  cmat[:, j].reshape(self.shape)))
        ei, ev = jla.eigh(mat)
        return cmat@ev@(1/jnp.sqrt(jnp.abs(ei))[:, None]*ev.T)

    def lb_test(self, x, f):
        ret = 0
        rgradf = jax.jit(lambda x: self.proj(
            x,
            self.inv_g_metric(x, grad(f)(x))))
        obs = make_tangent_basis(self, x)
        for i in range(self.dim):
            obsi = obs[:, i].reshape(self.shape)
            tmp = jvp(rgradf, (x,), (obsi,))
            nxi = tmp[1] + self.gamma(x, obsi, tmp[0])
            ret += self.inner(x, obsi, nxi)
        return ret

    print(lb_test(spd, x, f))
    print(spd.laplace_beltrami(x, egradf(x), ehessf))


if __name__ == '__main__':
    test_spd()

