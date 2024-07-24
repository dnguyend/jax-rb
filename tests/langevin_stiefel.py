""" test riemannian langevin for stiefel manifolds
"""

from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random, vmap, jit
"""
"""
from jax.scipy.linalg import expm
import jax_rb.manifolds.stiefel as stm

from jax_rb.utils.utils import (sym, grand)
import jax_rb.simulation.simulator as sim
import jax_rb.simulation.global_manifold_integrator as gmi


jax.config.update("jax_enable_x64", True)


def sqr(x):
    return x@x


def cz(mat):
    return jnp.max(jnp.abs(mat))


class stiefel_polar_retraction():
    def __init__(self, mnf):
        self.mnf = mnf

    def retract(self, x, v):
        """rescaling :math:`x+v` to be on the manifold
        """
        u, _, vt = jla.svd(x+v, full_matrices=False)
        return u@vt

    def drift_adjust(self, x):
        n, d, alp1 = self.mnf.shape[0], self.mnf.shape[1], self.mnf.alpha[1]
        return -0.5*(n-d+0.5*(d-1)/alp1)*x
    

def uniform_sampling(key, shape, pay_off, n_samples):
    """ Sample the manifold uniformly
    """
    x_all, key = grand(key, (shape[0], shape[1], n_samples))

    def do_one_point(seq):
        # ei, ev = jla.eigh(seq.T@seq)
        # return pay_off(seq@ev@((1/jnp.sqrt(ei))[:, None]*ev.T))
        u, _, vt = jla.svd(seq, full_matrices=False)
        return pay_off(u[:, :shape[0]]@vt)

    s = jax.vmap(do_one_point, in_axes=2)(x_all)
    return jnp.nanmean(s)


def test_stiefel_langevin_von_mises_fisher(key, stf, kp, F, func):
    # test Langevin on stiefel with vfunc = e^{-\frac{1}{2}v^T\Lambda v}
    # jax.config.update('jax_default_device', jax.devices('cpu')[0])
    print("Doing Stiefel von Mises Fisher (n, d)=%s alpha=%s" % (str(stf.shape), str(stf.alpha)))
        
    @partial(jit, static_argnums=(0,))
    def log_v(_, x):
        return kp*jnp.trace(F.T@x)

    @partial(jit, static_argnums=(0,))    
    def grad_log_v(mnf, x):
        return kp*mnf.proj(x, mnf.inv_g_metric(x, F))
    
    x, key = stf.rand_point(key)
    eta, key = stf.rand_vec(key, x)

    # print(jax.jvp(lambda x: log_v(stf, x), (x,), (eta,))[1])
    # print(stf.inner(x, grad_log_v(stf, x), eta))

    pay_offs = [None, func]

    x_0, key = stf.rand_point(key)
    key, sk = random.split(key)
    t_final = 5.
    n_path = 10000
    n_div = 500
    d_coeff = .5

    wiener_dim = stf.shape[0]*stf.shape[1]
    # crtr = cayley_se_retraction(se)

    # rbrownian_ito_langevin_move(mnf, x, unit_move, scale, grad_log_v)    
    ret_rtr1 = sim.simulate(x_0,
                            lambda x, unit_move, scale: gmi.ito_move_with_drift(
                                stf, x, unit_move, scale, 0.5*grad_log_v(stf, x)),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("ito langevin %.3f" % jnp.nanmean(ret_rtr1[0]))

    ret_rtr2 = sim.simulate(x_0,
                            lambda x, unit_move, scale: gmi.stratonovich_move_with_drift(
                                stf, x, unit_move, scale, 0.5*grad_log_v(stf, x)),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("stratonovich langevin %.3f" % jnp.nanmean(ret_rtr2[0]))

    ret_rtr3 = sim.simulate(x_0,
                            lambda x, unit_move, scale: gmi.geodesic_move_with_drift(
                                stf, x, unit_move, scale, 0.5*grad_log_v(stf, x)),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("geodesic langevin %.3f" % jnp.nanmean(ret_rtr3[0]))

    n_samples = 1000**2
    ret_spl =  uniform_sampling(key, stf.shape,
                              lambda x: pay_offs[1](x)*jnp.exp(log_v(None, x)),
                              n_samples)

    ret_spl_0 =  uniform_sampling(key, stf.shape,
                                lambda x: jnp.exp(log_v(None, x)),
                                n_samples)
    
    print("stiefel uniform sampling with density %.3f" % (ret_spl/ret_spl_0))
    # import scipy.special as ss
    # print(jnp.sqrt(2)*ss.iv(1, 1)/ss.iv(.5, 1)*ss.gamma(1.5))


def test_stiefel_langevin_bingham(key, stf, A, func):
    # test Langevin on stiefel with vfunc = e^{-\frac{1}{2}v^T\Lambda v}
    # jax.config.update('jax_default_device', jax.devices('cpu')[0])
    @partial(jit, static_argnums=(0,))
    def log_v(_, x):
        return jnp.trace(x.T@A@x)

    @partial(jit, static_argnums=(0,))    
    def grad_log_v(mnf, x):
        return mnf.proj(x, mnf.inv_g_metric(x, 2*A@x))

    print("Doing Bingham (n, d)=%s alpha=%s" % (str(stf.shape), str(stf.alpha)))
    
    # x, key = stf.rand_point(key)
    # eta, key = stf.rand_vec(key, x)

    # print(jax.jvp(lambda x: log_v(stf, x), (x,), (eta,))[1])
    # print(stf.inner(x, grad_log_v(stf, x), eta))

    pay_offs = [None, func]

    x_0, key = stf.rand_point(key)
    key, sk = random.split(key)
    t_final = 5.
    n_path = 10000
    n_div = 500
    d_coeff = .5

    wiener_dim = stf.shape[0]*stf.shape[1]
    # crtr = cayley_se_retraction(se)

    # rbrownian_ito_langevin_move(mnf, x, unit_move, scale, grad_log_v)    
    ret_rtr1 = sim.simulate(x_0,
                            lambda x, unit_move, scale: gmi.ito_move_with_drift(
                                stf, x, unit_move, scale, 0.5*grad_log_v(stf, x)),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("ito langevin %.3f" % jnp.nanmean(ret_rtr1[0]))

    ret_rtr2 = sim.simulate(x_0,
                            lambda x, unit_move, scale: gmi.stratonovich_move_with_drift(
                                stf, x, unit_move, scale, 0.5*grad_log_v(stf, x)),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("stratonovich langevin %.3f" % jnp.nanmean(ret_rtr2[0]))

    ret_rtr3 = sim.simulate(x_0,
                            lambda x, unit_move, scale: gmi.geodesic_move_with_drift(
                                stf, x, unit_move, scale, 0.5*grad_log_v(stf, x)),
                            pay_offs[0],
                            # lambda x: x[1, -1]*x[1, -1],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("geodesic langevin %.3f" % jnp.nanmean(ret_rtr3[0]))

    n_samples = 1000**2
    ret_spl =  uniform_sampling(key, stf.shape,
                              lambda x: pay_offs[1](x)*jnp.exp(log_v(None, x)),
                              n_samples)

    ret_spl_0 =  uniform_sampling(key, stf.shape,
                                lambda x: jnp.exp(log_v(None, x)),
                                n_samples)
    
    print("stiefel uniform sampling with density %.3f" % (ret_spl/ret_spl_0))

    
def test_all_stiefel_von_mises_fisher():
    n = 3
    d = 1
    alp = jnp.array([1, .6])
    key = random.PRNGKey(0)
    
    stf = stm.RealStiefelAlpha((n, d), alp)

    # F, key = stf.rand_point(key)
    kp = 1.2
    F, key = stf.rand_point(key)
    test_stiefel_langevin_von_mises_fisher(
        key, stf, kp, F,
        lambda x: jnp.sqrt(jnp.abs(1-jnp.trace(F.T@x)**2)))
    # print(jnp.sqrt(2)*ss.iv(1, 1)/ss.iv(.5, 1)*ss.gamma(1.5))

    n = 5
    d = 3
    alp = jnp.array([1, .6])
    key = random.PRNGKey(0)
    
    stf = stm.RealStiefelAlpha((n, d), alp)

    # F, key = stf.rand_point(key)
    kp = 1.2
    F, key = stf.rand_point(key)
            
    test_stiefel_langevin_von_mises_fisher(key, stf, kp, F, lambda x: jnp.sqrt(jnp.abs(1-jnp.trace(F.T@x)**2)))

    test_stiefel_langevin_von_mises_fisher(key, stf, kp, F,
                                           lambda x: jnp.sum(jnp.abs(x)))


def gen_sym_traceless(key, n):
    A, key = grand(key, (n, n))
    return sym(A) - jnp.trace(A)/n*jnp.eye(n), key


def test_all_bingham():
    n = 3
    d = 1
    alp = jnp.array([1, .6])
    key = random.PRNGKey(0)
    
    stf = stm.RealStiefelAlpha((n, d), alp)

    A, key = gen_sym_traceless(key, n)
    test_stiefel_langevin_bingham(
        key, stf, A,
        lambda x: jnp.sum(jnp.abs(x)))
    # print(jnp.sqrt(2)*ss.iv(1, 1)/ss.iv(.5, 1)*ss.gamma(1.5))

    n = 5
    d = 3
    alp = jnp.array([1, .6])
    key = random.PRNGKey(0)

    stf = stm.RealStiefelAlpha((n, d), alp)
    A, key = gen_sym_traceless(key, n)
    test_stiefel_langevin_bingham(
        key, stf, A,
        lambda x: jnp.sum(jnp.abs(x)))

    test_stiefel_langevin_bingham(
        key, stf, A,
        lambda x: jnp.sum(jnp.abs(x)*(A@jnp.abs(x))))

    n = 7
    d = 3
    alp = jnp.array([1, .6])
    key = random.PRNGKey(0)

    stf = stm.RealStiefelAlpha((n, d), alp)
    A, key = gen_sym_traceless(key, n)
    test_stiefel_langevin_bingham(
        key, stf, A,
        lambda x: jnp.sum(jnp.abs(x)))

    test_stiefel_langevin_bingham(
        key, stf, A,
        lambda x: jnp.sum(jnp.abs(x)*(A@jnp.abs(x))))


def drift_adjust_verify(self, x, sigma, wiener_dim):
    """return the adjustment :math:`\\mu_{adj}`
    so that :math:`\\mu + \\mu_{adj} = \\mu_{\\mathfrak{r}}`
    """
    def sqt(a):
        return a.T@a

    return -0.5*x@jnp.sum(vmap(lambda seq:
                               sqt(self.proj(x, sigma(x, seq.reshape(x.shape)))))(jnp.eye(wiener_dim)),
                          axis=0)

    
def test_polar_retract_adjust():
    n = 7
    d = 3
    alp = jnp.array([1, .6])
    key = random.PRNGKey(0)
    stf = stm.RealStiefelAlpha((n, d), alp)
    print("Doing Stiefel Polar retract for Bingham (n, d)=%s alpha=%s" % (str(stf.shape), str(stf.alpha)))
    @partial(jit, static_argnums=(0,))
    def log_v(_, x):
        return jnp.trace(x.T@A@x)

    @partial(jit, static_argnums=(0,))    
    def grad_log_v(mnf, x):
        return mnf.proj(x, mnf.inv_g_metric(x, 2*A@x))

    x, key = stf.rand_point(key)

    # mu2 = -0.5*(n-d+0.5*(d-1)/alp[1])*x
    prtr = stiefel_polar_retraction(stf)

    mu1 = drift_adjust_verify(stf, x, stf.sigma, n*d)
    mu2 = prtr.drift_adjust(x)
    # print(mu2-mu1)

    A, key = gen_sym_traceless(key, n)

    x_0, key = stf.rand_point(key)
    pay_offs = [None, lambda x: jnp.sum(jnp.abs(x))]

    key, sk = random.split(key)
    t_final = 5.
    n_path = 10000
    n_div = 500
    d_coeff = .5

    wiener_dim = stf.shape[0]*stf.shape[1]

    test_stiefel_langevin_bingham(
        key, stf, A,
        pay_offs[1])

    ret_rtr = sim.simulate(x_0,
                           lambda x, unit_move, scale: prtr.retract(
                               x, stf.proj(x, stf.sigma(x, unit_move.reshape(x.shape)*scale**.5
                                                        + scale*(
                                                            stf.ito_drift(x)
                                                            + 0.5*grad_log_v(stf, x))))),
                           pay_offs[0],
                           pay_offs[1],
                           [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print("Polar adjust %.3f" % jnp.nanmean(ret_rtr[0]))


if __name__ == '__main__':
    test_all_stiefel_von_mises_fisher()    
    test_all_bingham()
    test_polar_retract_adjust()
