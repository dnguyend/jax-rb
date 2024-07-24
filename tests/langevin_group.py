""" test riemannian langevin for SO and SE
"""


from functools import partial

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random, vmap, jit
import jax_rb.manifolds.so_left_invariant as som
import jax_rb.manifolds.se_left_invariant as sem

from jax_rb.utils.utils import (rand_positive_definite, sym, vcat, grand)
import jax_rb.simulation.simulator as sim
import jax_rb.simulation.matrix_group_integrator as mi


jax.config.update("jax_enable_x64", True)


def sqr(x):
    return x@x


def cz(mat):
    return jnp.max(jnp.abs(mat))


class cayley_so_retraction():
    """Cayley retraction of a matrix Lie group
    this is the most general, and not efficient implementation
    for each lie group, we should have a custom implementation of this
    """
    def __init__(self, mnf):
        self.mnf = mnf

    def retract(self, x, v):
        """rescaling :math:`x+v` to be on the manifold
        """
        ixv = x.T@v
        return x + x@jla.solve(jnp.eye(ixv.shape[0]) - 0.5*ixv, ixv)

    def inverse_retract(self, x, y):
        u = x.T@y
        n = self.mnf.shape[0]
        return 2*x@jla.solve(jnp.eye(n)+u, u-jnp.eye(n))

    def drift_adjust(self, x, driver_dim):
        """return the adjustment :math:`\\mu_{adj}`
        so that :math:`\\mu + \\mu_{adj} = \\mu_{\\mathfrak{r}}`
        """
        return -0.5*jnp.sum(vmap(lambda seq:
                                 x@sqr(self.mnf.sigma_id(seq.reshape(x.shape)))
                                 )(jnp.eye(driver_dim)),
                            axis=0)


def uniform_sample(key, shape, pay_off, n_samples):
    """ Sample the manifold uniformly
    """
    x_all, key = grand(key, (shape[0], shape[1], n_samples))

    def do_one_point(seq):
        # ei, ev = jla.eigh(seq.T@seq)
        # return pay_off(seq@ev@((1/jnp.sqrt(ei))[:, None]*ev.T))
        u, _, vt = jla.svd(seq)
        return pay_off(u[:, :shape[0]]@vt)

    s = jax.vmap(do_one_point, in_axes=2)(x_all)
    return jnp.nanmean(s)


def test_langevin_so():
    # test Langevin on se(n) with vfunc = e^{-\frac{1}{2}v^T\Lambda v}
    # jax.config.update('jax_default_device', jax.devices('cpu')[0])
    n = 4
    so_dim = n*(n-1)//2

    lbd = 2.1*jnp.arange(1, so_dim+1)

    def log_v(_, x):
        return -jnp.sum(x[jnp.triu_indices(n,1)].reshape(-1)**2*lbd)

    def grad_log_v(mnf, x):
        idx = jnp.triu_indices(n,1)
        return mnf.proj(x, mnf.inv_g_metric(
            x,
            jnp.zeros_like(x).at[idx].set(-2*lbd*x[idx].reshape(-1))))

    key = random.PRNGKey(0)

    metric_mat, key = rand_positive_definite(key, so_dim, (.1, 30.))

    print("Doing SO")

    # metric_mat = jnp.eye(se_dim)
    so = som.SOLeftInvariant(n, metric_mat)
    crtr = cayley_so_retraction(so)
    # x, key = so.rand_point(key)
    # eta, key = so.rand_vec(key, x)
    # x1 = crtr.retract(x, eta)
    # eta1 = crtr.inverse_retract(x, x1)
    # print(cz(eta1-eta))

    # pay_offs = [None, lambda x: jnp.sqrt(jnp.sum(x*x))]
    pay_offs = [None, lambda x: jnp.sqrt(1+jnp.sum(jnp.abs(x)))]
    # lbd1 = jnp.arange(1, n**2+1)
    lbd1, key = grand(key, (n**2,))
    pay_offs = [None, lambda x: jnp.sqrt(1+jnp.sum(jnp.abs(lbd1*x.reshape(-1))))]

    x_0 = jnp.eye(n)

    key, sk = random.split(key)
    t_final = 40.
    # t_final = 1.5
    n_path = 1000
    n_div = 1000
    d_coeff = .5

    wiener_dim = n**2

    ret_rtr = sim.simulate(x_0,
                           lambda x, unit_move, scale: crtr.retract(
                               x,
                               x@so.sigma_id(unit_move.reshape(x.shape))*scale**.5
                               + 0.5*grad_log_v(so, x)*scale),
                           pay_offs[0],
                           pay_offs[1],
                           [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("SO Cayley retract %.3f" % jnp.nanmean(ret_rtr[0]))

    # The more effective sample
    n_samples = 1000**2
    ret_denom = uniform_sample(
        key, so.shape,
        lambda x: jnp.exp(log_v(so, x)),
        n_samples)
    
    ret_num = uniform_sample(
        key, so.shape,
        lambda x: pay_offs[1](x)*jnp.exp(log_v(so, x)),
        n_samples)
    print("SO sampling with density %.3f" % (ret_num/ret_denom))


def test_langevin_se():
    # test Langevin on se(n) with vfunc = e^{-\frac{1}{2}v^T\Lambda v}
    # jax.config.update('jax_default_device', jax.devices('cpu')[0])
    n = 3
    # lbd = 0.5*jnp.arange(1, n+1)
    lbd = 0.5*jnp.ones(n)
    # lbd = jnp.array([1., 100.])
    lbd = 10.*jnp.arange(1, n+1)

    @partial(jit, static_argnums=(0,))    
    def log_v(_, x):
        return -0.5*jnp.sum(x[:-1, -1]*lbd*x[:-1, -1])

    @partial(jit, static_argnums=(0,))
    def grad_log_v(mnf, x):
        return mnf.proj(x, mnf.inv_g_metric(
            x,
            jnp.zeros_like(x).at[:-1, -1].set(-lbd*x[:-1, -1])))

    key = random.PRNGKey(0)

    se_dim = n*(n+1)//2
    n1 = n+1
    metric_mat, key = rand_positive_definite(key, se_dim, (.1, 30.))

    # convergent seems to be to same metric, but different rate
    
    # metric_mat = jnp.eye(se_dim)
    # metric_mat = metric_mat.at[0, 0].set(1.)
    se = sem.SELeftInvariant(n, metric_mat)
    # x, key = se.rand_point(key)
    # eta, key = se.rand_vec(key, x)

    # print(jax.jvp(lambda x: log_v(se, x), (x,), (eta,))[1])
    # print(se.inner(x, grad_log_v(se, x), eta))

    # pay_offs = [None, lambda x: jnp.sqrt(jnp.sum(x[:-1, -1]**2))]

    # pay_offs = [None, lambda x: jnp.sum(jnp.abs(x[:-1, -1]))]

    # pay_offs = [None, lambda x: jnp.sqrt(jnp.sum(x*x))]
    print("Test SE with n=%d expectation of sum |x|" % (n)) 
    pay_offs = [None, lambda x: jnp.sum(jnp.abs(x))]

    x_0 = jnp.eye(n1)
    key, sk = random.split(key)
    t_final = 100.
    n_path = 5000
    n_div = 1000
    d_coeff = .5

    wiener_dim = n1**2
    ret_rtr1 = sim.simulate(x_0,
                            lambda x, unit_move, scale: mi.ito_move_with_drift(
                                se, x, unit_move, scale, 0.5*jla.solve(x, grad_log_v(se, x))),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("Ito Langevin %.3f" % jnp.nanmean(ret_rtr1[0]))

    ret_rtr2 = sim.simulate(x_0,
                            lambda x, unit_move, scale: mi.stratonovich_move_with_drift(
                                se, x, unit_move, scale, 0.5*jla.solve(x, grad_log_v(se, x))),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("Stratonovich Langevin %.3f" % jnp.nanmean(ret_rtr2[0]))


    ret_rtr3 = sim.simulate(x_0,
                            lambda x, unit_move, scale: mi.geodesic_move_with_drift(
                                se, x, unit_move, scale, 0.5*jla.solve(x, grad_log_v(se, x))),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],                            
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("Geodesic 2nd order langevin %.3f" % jnp.nanmean(ret_rtr3[0]))
    
    
    def se_sample(key, shape, pay_off, n_samples):
        """ Sample the manifold uniformly on the sphere
        and with the 
        """
        x_all, key = grand(key, (shape[0]-1, shape[1], n_samples))

        def do_one_point(seq):
            # ei, ev = jla.eigh(seq.T@seq)
            # return pay_off(seq@ev@((1/jnp.sqrt(ei))[:, None]*ev.T))
            u, _, vt = jla.svd(seq[:, :-1])
            x = vcat(jnp.concatenate(
                [u@vt, seq[:, -1][:, None]], axis=1),
                     jnp.zeros((1, shape[1])).at[0, -1].set(1.))
            return pay_off(x)*jnp.exp(log_v(se, x)+0.5*jnp.sum(x[:-1, -1]**2))

        s = jax.vmap(do_one_point, in_axes=2)(x_all)
        return jnp.nanmean(s)

    n_samples = 1000**2    
    ret_denom = se_sample(
        key, se.shape,
        lambda x: 1.,
        n_samples)
    ret_num = se_sample(
        key, se.shape,
        pay_offs[1],
        n_samples)
    
    print("uniform sampling with density %.3f" % (ret_num/ret_denom))


def test_langevin_se2():
    # jax.config.update('jax_default_device', jax.devices('cpu')[0])
    n = 4
    se_dim = n*(n+1)//2
    n1 = n+1
    
    lbd = 0.5*jnp.arange(1, n+1)
    lbd = 10.*jnp.ones(n)

    @partial(jit, static_argnums=(0,))    
    def log_v(_, x):
        return -0.5*jnp.sum(x[:-1, -1]*lbd*x[:-1, -1])

    @partial(jit, static_argnums=(0,))    
    def grad_log_v(mnf, x):
        return mnf.proj(x, mnf.inv_g_metric(
            x,
            jnp.zeros_like(x).at[:-1, -1].set(-lbd*x[:-1, -1])))

    key = random.PRNGKey(0)

    # metric_mat, key = rand_positive_definite(key, se_dim, (.1, 30.))
    A, key = grand(key, (n*n1,n*n1))
    A = sym(A@A.T)
    # convergent seems to be to same metric, but different rate
    
    metric_mat = jnp.eye(se_dim)
    # metric_mat = metric_mat.at[0, 0].set(1.)
    se = sem.SELeftInvariant(n, metric_mat)
    # x, key = se.rand_point(key)
    # eta, key = se.rand_vec(key, x)

    # print(jax.jvp(lambda x: log_v(se, x), (x,), (eta,))[1])
    # print(se.inner(x, grad_log_v(se, x), eta))
    print("Test SE n=%d expectation of  |x^TAx|^(1/2) for a positive definite matrix A" % (n))
    
    pay_offs = [None, lambda x: jnp.sqrt(jnp.abs(jnp.sum(x[:-1, :].reshape(-1)*(A@x[:-1, :].reshape(-1)))))]
    # pay_offs = [None, lambda x: jnp.sqrt(jnp.sum(x*x*jnp.arange(1, n1+1)[None, :]))]
    # pay_offs = [None, lambda x: jnp.sum(x[0, :-1]*x[:-1, -1])]

    x_0 = jnp.eye(n1)
    key, sk = random.split(key)
    t_final = 50.
    n_path = 5000
    n_div = 1000
    d_coeff = .5

    wiener_dim = n1**2
    
    ret_rtr1 = sim.simulate(x_0,
                            lambda x, unit_move, scale: mi.ito_move_with_drift(
                                se, x, unit_move, scale, 0.5*jla.solve(x, grad_log_v(se, x))),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("Ito Langevin %.3f" % jnp.nanmean(ret_rtr1[0]))


    ret_rtr2 = sim.simulate(x_0,
                            lambda x, unit_move, scale: mi.stratonovich_move_with_drift(
                                se, x, unit_move, scale, 0.5*jla.solve(x, grad_log_v(se, x))),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("Stratonovich Langevin %.3f" % jnp.nanmean(ret_rtr2[0]))


    ret_rtr3 = sim.simulate(x_0,
                            lambda x, unit_move, scale: mi.geodesic_move_with_drift(
                                se, x, unit_move, scale, 0.5*jla.solve(x, grad_log_v(se, x))),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("Geodesic 2nd order langevin %.3f" % jnp.nanmean(ret_rtr3[0]))
    
    
    def se_sample(key, shape, pay_off, n_samples):
        """ Sample the manifold uniformly on the sphere
        and with the 
        """
        x_all, key = grand(key, (shape[0]-1, shape[1], n_samples))

        def do_one_point(seq):
            # ei, ev = jla.eigh(seq.T@seq)
            # return pay_off(seq@ev@((1/jnp.sqrt(ei))[:, None]*ev.T))
            u, _, vt = jla.svd(seq[:, :-1])
            x = vcat(jnp.concatenate(
                [u@vt, seq[:, -1][:, None]], axis=1),
                     jnp.zeros((1, shape[1])).at[0, -1].set(1.))
            return pay_off(x)*jnp.exp(log_v(se, x)+0.5*jnp.sum(x[:-1, -1]**2))
        #return jnp.sqrt(3+jnp.sum(x[:-1, -1]**2))*jnp.exp(log_v(se, x)+0.5*jnp.sum(x[:-1, -1]**2))

        s = jax.vmap(do_one_point, in_axes=2)(x_all)
        # ret = []
        # for i in range(x_all.shape[2]):
        #    ret.append(do_one_point(x_all[:, :, i]))
        # s = jnp.array(ret)
        return jnp.nanmean(s)

    n_samples = 1000**2    
    
    ret_denom = se_sample(
        key, se.shape,
        lambda x: 1.,
        n_samples)
    """
    ret_num = se_sample(
        key, se.shape,
        lambda x: pay_offs[1](x),
        n_path*500)
    """
    ret_num = se_sample(
        key, se.shape,
        # lambda x: x[1, -1]*x[1, -1],
        pay_offs[1],
        # lambda x: pay_offs[1](x) - jnp.sqrt(3+jnp.sum(x[:-1, -1]**2)),
        n_samples)
    
    print("uniform sampling with density %.3f" % (ret_num/ret_denom))
    
    
if __name__ == '__main__':
    test_langevin_so()
    test_langevin_se()
    test_langevin_se2()            
