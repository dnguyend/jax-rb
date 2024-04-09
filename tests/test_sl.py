import jax
import jax.numpy as jnp
from jax import random, jvp, grad

from jax_rb.manifolds.sl_left_invariant import SLLeftInvariant
from jax_rb.utils.utils import (grand, rand_positive_definite)

jax.config.update("jax_enable_x64", True)


def test_sl():
    key = random.PRNGKey(0)
    n = 3

    metric_mat, key = rand_positive_definite(key, n**2-1)
    mnf = SLLeftInvariant(n, metric_mat)
    print(mnf.v0)

    x, key = mnf.rand_point(key)

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

    print(mnf.inner(x,
                    mnf.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[1, 2].set(1)),
                    mnf.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[1, 2].set(1))))
    
    print(mnf.inner(x,
                    mnf.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[0, 2].set(1)),
                    mnf.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[1, 2].set(1))))
    

    print(jvp(lambda x: mnf.inner(x, vb, vb), (x,), (va,))[1])
    print(2*mnf.inner(x, vb, jvp(lambda x: mnf.proj(x, vb), (x,), (va,))[1]
                      + mnf.gamma(x, va, vb)))

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
        for i in range(p):
            for j in range(p):
                vij = self._mat_apply(i_sqrt_g_mat,
                                      zr.at[i, j].set(1.))

                vv += self.gamma(jnp.eye(n), vij, vij)
                vv += jvp(lambda x: mnf.sigma(x, zr.at[i, j].set(1.)),
                          (x,), (vij,))[1]
        return -0.5*vv
    
    print(check_v0(mnf))
    print(mnf.v0)    

    # now test Laplacian

    f1, key = grand(key, (n, n))
    f2, key = grand(key, (n*n, n*n))
    f3, key = grand(key, (n*n, n*n))    

    @jax.jit    
    def f(U):
        return jnp.sum(f1*U) + jnp.sum(U.reshape(-1)*(f2@U.reshape(-1))) \
            + jnp.sum(U.reshape(-1)*(f2@U.reshape(-1)))**2

    egradf = jax.jit(jax.grad(f))

    @jax.jit
    def ehessf(U, omg):
        return jvp(egradf, (U,), (omg,))[1]

    def lb_test(self, x, f):
        n = self.shape[0]
        ret = 0
        rgradf = jax.jit(lambda x: self.proj(
            x,
            self.inv_g_metric(x, grad(f)(x))))
        for i in range(n):
            for j in range(n):
                # vij = self.v_ij(x, i, j)
                vij = self.left_invariant_vector_field(x, jnp.zeros(mnf.shape).at[i, j].set(1.))
                
                tmp = jvp(rgradf, (x,), (vij,))
                nxi = tmp[1] + self.gamma(x, vij, tmp[0])
                ret += self.inner(x, vij, nxi)
        return ret
    print(lb_test(mnf, x, f))
    print(mnf.laplace_beltrami(x, egradf(x), ehessf))


def jk():    
    print(jvp(lambda x: mnf.inner(x, omg1, omg1), (x,), (omg2,))[1])
    print(2*mnf.inner(x, omg1, mnf.gamma(x, omg1, omg2)))

    def check_v0(self):
        # compute as derivative of projection
        p, i_sqrt_lu_mat = self.shape[0], self._i_sqrt_g_mat
        vv = jnp.zeros((p, p))
        zr = jnp.zeros((p, p))
        for i in range(p):
            for j in range(p):
                vij = self._mat_apply(i_sqrt_lu_mat,
                                      zr.at[i, j].set(1.))

                vv += self.gamma(jnp.eye(n), vij, vij)
                vv += jvp(lambda x: mnf.sigma(x, zr.at[i, j].set(1.)),
                          (x,), (vij,))[1]
        return -0.5*vv

    print(check_v0(mnf))
    
    def make_drift1(self, x):
        """make the drift at identity
        """
        p, i_sqrt_g_mat = self.shape[0], self._i_sqrt_g_mat
        vv = jnp.zeros((p, p))
        zr = jnp.zeros((p, p))
        gid = jnp.eye(p)

        def proj_sigma(x, omg):
            # return self.proj(x, mnf.sigma(x, omg))
            return x@self._mat_apply(
                i_sqrt_g_mat,
                self._lie_algebra_proj(omg))

        for i in range(p):
            for j in range(p):
                eij = zr.at[i, j].set(1.)
                vij = x@self._mat_apply(i_sqrt_g_mat,
                                        self._lie_algebra_proj(eij))

                vv += self.gamma(x, vij, vij)
                
        return -0.5*vv
    
    
    print(make_drift1(mnf, x) - mnf.ito_drift(x))

    def make_id_drift1(self):
        """make the drift at identity
        """
        p = self.p
        drft = jnp.zeros((p, p))
        for i in range(p):
            for j in range(p):
                eij = jnp.zeros((p, p)).at[i, j].set(1.)
                xi = asym(eij)
                eta = asym(inv_lf_opt(self._lu_mat, eij))
                drft += asym(eij)@asym(inv_lf_opt(self._lu_mat, eij)) +\
                    0*0.5*inv_lf_opt(self._lu_mat, Lie(xi, lf_opt(self._lu_mat, eta))
                             + Lie(eta, lf_opt(self._lu_mat, xi)))

        # self.gamma(jnp.eye(n), asym(eij),
        # asym(inv_lf_opt(self._lu_mat, eij)))
        return 0.5*drft

    print(mnf.make_id_drift() - make_id_drift1(mnf))

    def make_id_drift2(self):
        """make the drift at identity
        """
        n, i_sqrt_lu_mat = self.n, self._i_sqrt_lu_mat
        drft = jnp.zeros((n, n))
        for i in range(n):
            for j in range(n):
                eij = jnp.zeros((n, n)).at[i, j].set(1.)
                xi = 1/jnp.sqrt(2)*lf_opt(i_sqrt_lu_mat, asym(eij))
                drft += 0.5*xi@xi
        # self.gamma(jnp.eye(n), asym(eij),
        # asym(inv_lf_opt(self._lu_mat, eij)))
        return drft

    print(mnf.make_id_drift() - make_id_drift1(mnf))
    

    def rbs_move(self, x, dw, dt):
        """ Stratonovich version
        """
        bdr = self.rbrownian_ito_drift(x)
        vtmp = self.sigma(x, dw) + bdr*dt
        v = self.proj(x, vtmp)
        vb  = self.proj(self.retract(x, v), vtmp)
        return self.retract(x, 0.5*(v+vb))
    


def test_so_n_inertia():
    from jax import random, jvp, grad
    key = random.PRNGKey(0)
    jax.config.update("jax_enable_x64", True)

    n = 4

    i_mat, key = gen_inertia_matrix(key, n)
    lu_mat = jnp.diag(i_mat[jnp.triu_indices(n, 1)])
    mnf = SOLeftInvariant(n, lu_mat)
    imf = SOInertiaMetric(n, i_mat)

    x, key = imf.randpoint(key)
    print(x@x.T - jnp.eye(n))
    y, key = imf.randpoint(key)

    v, key = imf.randvec(key, x)
    print(sym(x.T@v))

    # check metric compatibility
    va, key = imf.randvec(key, x)
    vb, key = imf.randvec(key, x)
    omg, key = imf.rand_amb(key)
    omg0, key = imf.rand_amb(key)

    omg1 = imf.g_metric(x, omg)
    omg1a = mnf.g_metric(x, omg)
    print(omg1 - omg1a)
    
    omg2 = imf.inv_g_metric(x, omg1)
    print(omg2 - omg)
    print(jnp.sum(va*omg1) - mnf.inner(x, va, omg))

    pomg = mnf.proj(x, omg)
    pomga = imf.proj(x, omg)
    print(pomg - pomga)
    # print(jnp.sum(x*pomg))
    print(imf.inner(x, omg, va) - imf.inner(x, mnf.proj(x, omg), va))

    print(jvp(lambda x: imf.inner(x, vb, vb), (x,), (va,))[1])
    print(2*imf.inner(x, vb, 
                      jvp(lambda x: imf.proj(x, vb), (x,), (va,))[1]
                      + imf.gamma(x, va, vb)))

    D1 = jvp(lambda x: imf.proj(x, vb), (x,), (va,))[1] + imf.gamma(x, va, vb)
    print(sym(x.T@D1))

    # now test Laplacian

    f1, key = grand(key, (n, n))
    f2, key = grand(key, (n*n, n*n))
    f3, key = grand(key, (n*n, n*n))    

    @jax.jit    
    def f(U):
        return jnp.sum(f1*U) + jnp.sum(U.reshape(-1)*(f2@U.reshape(-1))) \
            + jnp.sum(U.reshape(-1)*(f2@U.reshape(-1)))**2

    egradf = jax.jit(jax.grad(f))

    @jax.jit
    def ehessf(U, omg):
        return jvp(egradf, (U,), (omg,))[1]

    def Lptest(self, x, f):
        p = self.p
        ret = 0
        rgradf = jax.jit(lambda x: self.proj(
            x,
            self.inv_g_metric(x, grad(f)(x))))
        for i in range(1, p):
            for j in range(i):
                vij = self.Vij(x, i, j)
                tmp = jvp(rgradf, (x,), (vij,))
                nxi = tmp[1] + self.gamma(x, vij, tmp[0])
                ret += self.inner(x, vij, nxi)
        return ret

    print(Lptest(imf, x, f))
    print(mnf.laplace_beltrami(x, egradf(x), ehessf))
    print(imf.laplace_beltrami(x, egradf(x), ehessf))

    # now Brownian simulation


if __name__ == '__main__':
    test_sl()
    
