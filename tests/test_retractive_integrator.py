"""test simulation with the
"""

import jax
import jax.numpy as jnp
from jax import random, jvp, vmap

from jax_rb.manifolds.diag_hypersurface import DiagHypersurface
from jax_rb.utils.utils import (grand)
import jax_rb.simulation.simulator as sim
import jax_rb.simulation.global_manifold_integrator as gmi
import jax_rb.simulation.retractive_integrator as rmi


class rescale_retraction():
    """the rescaling retraction on 
    diagonal constrained hypersurface
    """
    def __init__(self, mnf):
        self.mnf = mnf

    def retract(self, x, v):
        """rescaling :math:`x+v` to be on the hypersurface
        """
        val = self.mnf.cfunc(x+v)
        return (x+v)/val**(1/self.mnf.p)

    def hess(self, x, v):
        """hessian of the rescaling
        """
        p = self.mnf.p
        dvec = self.mnf.dvec
        return (1-p)*x*jnp.sum(dvec*x**(p-2)*v*v)

    def drift_adjust(self, sigma, x, t, driver_dim):
        """return the adjustment :math:`\\mu_{adj}`
        so that :math:`\\mu + \\mu_{adj} = \\mu_{\\mathfrak{r}}`
        """
        return -0.5*jnp.sum(vmap(lambda seq:
                                 self.hess(x, sigma(x, t, seq)))(jnp.eye(driver_dim)),
                            axis=0)


def test_retractive_integrator():
    n = 5
    p = 2
    key = random.PRNGKey(0)
    dvec, key = grand(key, (n,))
    dvec = dvec.at[-1].set(1.)

    mnf = DiagHypersurface(dvec, p)
    x, key = mnf.rand_point(key)
    # now test retract
    while True:
        q, key = mnf.rand_ambient(key)
        if mnf.cfunc(q) > 0:
            xq = mnf.approx_nearest(q)
            break
    print(f"test apprx nearest C(q)={mnf.cfunc(q)}, C(x)={mnf.cfunc(xq)}")

    # now tangent.
    xi, key = mnf.rand_vec(key, x)
    rtr = rescale_retraction(mnf)
    v = .01*xi
    x1 = rtr.retract(x, v)
    print(f"test retract C(rtr.retract(x, v)={mnf.cfunc(x1)}")

    def rt(t):
        return rtr.retract(x, t*v)

    def dr(t):
        p = rtr.mnf.p
        cft = rtr.mnf.cfunc(x+t*v)
        return -1/p*cft**(-1-1/p)*jnp.sum(rtr.mnf.grad_c(x+t*v)*v)*(x+t*v) \
            + cft**(-1/p)*v

    print("test deriv and hess of retract")
    print(jvp(rt, (.1,), (1.,))[1])
    print(dr(.1))
    print(jvp(dr, (0.,), (1.,))[1])
    print(rtr.hess(x, v))

    gsum = jnp.zeros(n)
    hsum = jnp.zeros(n)
    for i in range(n):
        nsg = mnf.proj(x, mnf.sigma(x, jnp.zeros(n).at[i].set(1.)))
        hsum += -rtr.hess(x, nsg)
        gsum += - mnf.gamma(x, nsg, nsg)
        # print(jnp.sum(mnf.grad_c(x)*(hsum-gsum)))

    print(f"test sum -gamma - ito drift={0.5*gsum - mnf.ito_drift(x)}")
    print(f"test adjusted ito is tangent={jnp.sum(mnf.grad_c(x)*(0.5*hsum+mnf.ito_drift(x)))}")

    # now test the equation.
    # test Brownian motion

    def new_sigma(x, _, dw):
        return mnf.proj(x, mnf.sigma(x, dw))

    def mu(x, _):
        return mnf.ito_drift(x)

    pay_offs = [lambda x, t: t*jnp.sum(x*jnp.arange(n)),
                lambda x: jnp.sum(x*x)]

    key, sk = random.split(key)
    t_final = 1.
    n_path = 1000
    n_div = 1000
    d_coeff = .5
    wiener_dim = n
    x_0 = jnp.zeros(n).at[-1].set(1)

    ret_geo = sim.simulate(x_0,
                           lambda x, unit_move, scale: gmi.geodesic_move(
                               mnf, x, unit_move, scale),
                           pay_offs[0],
                           pay_offs[1],
                           [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    ret_ito = sim.simulate(x_0,
                           lambda x, unit_move, scale: gmi.rbrownian_ito_move(
                               mnf, x, unit_move, scale),
                           pay_offs[0],
                           pay_offs[1],
                           [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    ret_str = sim.simulate(x_0,
                           lambda x, unit_move, scale: gmi.rbrownian_stratonovich_move(
                               mnf, x, unit_move, scale),
                           pay_offs[0],
                           pay_offs[1],
                           [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    ret_rtr = sim.simulate(x_0,
                           lambda x, unit_move, scale: rmi.retractive_move(
                               rtr, x, None, unit_move, scale, new_sigma, mu),
                           pay_offs[0],
                           pay_offs[1],
                           [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    ret_nrtr = sim.simulate(x_0,
                            lambda x, unit_move, scale: rmi.retractive_move_normalized(
                                rtr, x, None, unit_move, scale, new_sigma, mu),
                            pay_offs[0],
                            pay_offs[1],
                            [sk, t_final, n_path, n_div, d_coeff, wiener_dim])

    print(f"geo second order = {jnp.nanmean(ret_geo[0])}")
    print(f"Ito              = {jnp.nanmean(ret_ito[0])}")
    print(f"Stratonovich     = {jnp.nanmean(ret_str[0])}")
    print(f"Retractive       = {jnp.nanmean(ret_rtr[0])}")
    print(f"Retractive Norm. = {jnp.nanmean(ret_nrtr[0])}")


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    test_retractive_integrator()
