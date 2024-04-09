"""test basic functionalities of stiefel
"""
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random, jvp, grad

from jax_rb.manifolds.stiefel import RealStiefelAlpha
from jax_rb.utils.utils import (grand, sym)

jax.config.update("jax_enable_x64", True)


def test_stiefel():
    key = random.PRNGKey(0)

    n = 5
    p = 3

    al = jnp.array([.3, .5])
    stf = RealStiefelAlpha((n, p), al)
    x, key = stf.rand_point(key)
    print(x.T@x - jnp.eye(p))
    y, key = stf.rand_point(key)

    v, key = stf.rand_vec(key, x)
    print(sym(x.T@v))

    # check metric compatibility
    va, key = stf.rand_vec(key, x)
    vb, key = stf.rand_vec(key, x)
    omg, key = stf.rand_ambient(key)

    omg1 = stf.g_metric(x, omg)
    omg2 = stf.inv_g_metric(x, omg1)
    print(omg2 - omg)
    print(jnp.sum(va*omg1) - stf.inner(x, va, omg))

    pomg = stf.proj(x, omg)
    print(sym(x.T@pomg))
    print(stf.inner(x, omg, va) - stf.inner(x, stf.proj(x, omg), va))

    print(stf.gamma(x, va, vb))
    
    print(jvp(lambda x: stf.inner(x, vb, vb), (x,), (va,))[1])
    print(2*stf.inner(x, vb,
                      jvp(lambda x: stf.proj(x, vb), (x,), (va,))[1] + stf.gamma(x, va, vb)))

    print(stf.inner(x, stf.sigma(x, omg1), stf.sigma(x, omg))
          - jnp.sum(omg1*omg))
    
    
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

    check_ito_drift(stf, x)

    # now check laplace beltrami
    f1, key = grand(key, (n, n))
    f2, key = grand(key, (n, n))
    f2a, key = grand(key, (n, n))

    def f(q):
        return jnp.sum(f1*(q@q.T)) + jnp.trace(q@q.T@f2@q@q.T@f2a)

    egradf = jax.jit(jax.grad(f))
    
    def jvpf(q, omg):
        return jnp.sum(f1*(omg@q.T)) +jnp.sum(f1*(q@omg.T))\
            + jnp.trace(omg@q.T@f2@q@q.T@f2a) + jnp.trace(q@omg.T@f2@q@q.T@f2a) + jnp.trace(q@q.T@f2@omg@q.T@f2a) + jnp.trace(q@q.T@f2@q@omg.T@f2a)

    print(jvpf(x, omg) - jnp.sum(egradf(x)*omg))

    def ehessf02(q, omg1, omg2):
        return jnp.sum(f1*(omg1@omg2.T)) +jnp.sum(f1*(omg2@omg1.T))\
            + jnp.trace(omg1@omg2.T@f2@q@q.T@f2a) + jnp.trace(omg1@q.T@f2@omg2@q.T@f2a) + jnp.trace(omg1@q.T@f2@q@omg2.T@f2a) \
            + jnp.trace(omg2@omg1.T@f2@q@q.T@f2a) + jnp.trace(q@omg1.T@f2@omg2@q.T@f2a) + jnp.trace(q@omg1.T@f2@q@omg2.T@f2a) \
            + jnp.trace(omg2@q.T@f2@omg1@q.T@f2a) + jnp.trace(q@omg2.T@f2@omg1@q.T@f2a) + jnp.trace(q@q.T@f2@omg1@omg2.T@f2a) \
            + jnp.trace(omg2@q.T@f2@q@omg1.T@f2a) + jnp.trace(q@omg2.T@f2@q@omg1.T@f2a) + jnp.trace(q@q.T@f2@omg2@omg1.T@f2a)

    @jax.jit
    def ehessf(x, omg):
        return jvp(egradf, (x,), (omg,))[1]

    def make_proj_mat(stf, x):
        vshape = jnp.prod(jnp.array(stf.shape))
        mat = jnp.empty((vshape,  vshape))
        for i in range(vshape):
            mat = mat.at[:, i].set(
                stf.proj(x, jnp.zeros(vshape).at[i].set(1).reshape(stf.shape)).reshape(-1))
        return mat
    pmat = make_proj_mat(stf, x)
    print((pmat@omg.reshape(-1)).reshape(stf.shape) - stf.proj(x, omg))

    def  make_cmat(stf, x):
        pmat = make_proj_mat(stf, x)
        return jla.eigh(pmat)[1][:, -stf.dim:]

    def make_tangent_basis(self, x):
        d = stf.dim
        
        pmat = make_proj_mat(stf, x)
        cmat = jla.eigh(pmat)[1][:, -d:]
        mat = jnp.empty((d, d))
        for i in range(d):
            for j in range(d):
                mat = mat.at[i, j].set(self.inner(x, cmat[:, i].reshape(stf.shape),
                                                  cmat[:, j].reshape(stf.shape)))
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

    print(lb_test(stf, x, f))
    print(stf.laplace_beltrami(x, egradf(x), ehessf))



if __name__ == '__main__':
    test_stiefel()
