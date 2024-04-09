"""Module to compare with the heat kernel.
the heat kernel for upto 3d is implemented in utils.heat_kernels in the subfolder utils of the current folder.
Easiest way to run is to change to the repository containing the current folder then run
 run PYTHONPATH=., python sphere_heat_kernel.py
Install mpmath, which is needed in heat_kernels.py
"""

import jax
import jax.numpy as jnp
import jax.scipy.integrate as jsi
from jax import random

import numpy as np

import utils.heat_kernels as hkm
import jax_rb.simulation.simulator as sim
import jax_rb.simulation.global_manifold_integrator as mi

from jax_rb.manifolds.sphere import Sphere

jax.config.update("jax_enable_x64", True)


def test_1d():
    key = random.PRNGKey(0)
    key, sk = random.split(key)

    d = 1

    r = 1.2
    t_final = 1.1

    d0 = .4
    n_path = 500
    n_div = 200
    sph = Sphere(d+1, r)

    sph_intg = jnp.arccos(jnp.cos(
        jnp.sum(random.normal(sk, (n_div, n_path)), axis=0) *
        jnp.sqrt(t_final/(n_div)*2*d0)/sph.r))

    sph_sim0 = sim.simulate(sph.x0,
                         lambda x, unit_move, scale: mi.geodesic_move_normalized(
                             sph, x, unit_move, scale*d),
                         None,
                         lambda x: jnp.arccos(x[0]/sph.r),
                         (sk, t_final, n_path, n_div, d0, d+1))

    sph_sim1 = sim.simulate(sph.x0,
                         lambda x, unit_move, scale: mi.rbrownian_ito_move(
                             sph, x, unit_move, scale),
                         None,
                         lambda x: jnp.arccos(x[0]/sph.r),
                         (sk, t_final, n_path, n_div, d0, d+1))

    sph_sim2 = sim.simulate(sph.x0,
                         lambda x, unit_move, scale: mi.rbrownian_stratonovich_move(
                             sph, x, unit_move, scale),
                         None,
                         lambda x: jnp.arccos(x[0]/sph.r),
                         (sk, t_final, n_path, n_div, d0, d+1))

    print(jnp.nanmean(sph_intg), jnp.nanmean(sph_sim0[0]),
          jnp.nanmean(sph_sim1[0]),jnp.nanmean(sph_sim2[0]),
          )

    # second test

    r = 4.
    def fin(phi):
        # return phi**1.5 + phi**2.5
        # return phi**1.5 + phi**2.5 + phi**2.5
        return phi**2

    # new example
    sph = Sphere(d+1, r)
    sph_heat_kernel = np.trapz(
        np.array([fin(min(aa, 2*np.pi-aa))*hkm.thk1(0, min(aa, 2*np.pi-aa), t_final, d0/r**2)
                  for aa in np.arange(n_path+1)/n_path*np.pi]),
        dx=2*np.pi/n_path)

    # then change manifold range
    sph_sum = fin(jnp.arccos(jnp.cos(jnp.sum(random.normal(sk, (n_div, n_path)), axis=0)*jnp.sqrt(t_final/(n_div)*2*d0)/r)))
    
    # now random walk
    xtmp = random.normal(sk, (d, n_div, n_path))    
    xw = xtmp/jnp.sqrt(jnp.sum(xtmp**2, axis=0))[None, :]*jnp.sqrt(t_final/(n_div)*2*d0)
    sph_walk = jnp.mean(fin(jnp.arccos(jnp.cos(jnp.sum(xw, axis=1)/r))))

    x_0 = jnp.zeros(d+1).at[0].set(sph.r)
    sph_sim_geo = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.geodesic_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(sph.dist(x_0, x)/sph.r),
        (sk, t_final, n_path, n_div, d0, d+1))

    sph_sim_ito = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.rbrownian_ito_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(sph.dist(x_0, x)/sph.r),
        (sk, t_final, n_path, n_div, d0, d+1))

    sph_sim_strato = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.rbrownian_stratonovich_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(sph.dist(x_0, x)/sph.r),
        (sk, t_final, n_path, n_div, d0, d+1))

    print(f"heat_kernels={sph_heat_kernel}")
    print(f"sum_of_moves={jnp.mean(sph_sum)}")
    print(f"random walk={sph_walk}")

    print(f"geodesic={jnp.mean(sph_sim_geo[0])}")    
    print(f"ito={jnp.mean(sph_sim_ito[0])}")
    print(f"strato={jnp.mean(sph_sim_strato[0])}")


def test_2d():
    from scipy.integrate import dblquad
    
    key = random.PRNGKey(0)
    key, sk = random.split(key)

    d = 2
    r = 3
    t_final = 2.

    d0 = .4
    n_path = 1000
    n_div = 1000
    sph =Sphere(d+1, r)

    def fin(phi):
        return phi**2.5
    # return phi**1.5 + phi**2.5
    # return phi**1.5 + phi**2.5 + phi**2.5

    sph_heat_kernel = jsi.trapezoid(
        np.array([hkm.k2(phi, t_final*d0/r**2)*(np.sin(phi))*2*np.pi*fin(phi)
                  for phi in np.arange(n_path+1)/n_path*np.pi]),
        dx=np.pi/n_path)

    # compute the 2d heat kernel by integrating the 3d
    
    ss = dblquad(lambda u, phi: hkm.k3(np.arccos(np.sin(u)*np.cos(phi/2)), t_final*d0/sph.r**2/4)*(np.sin(phi))**2*2*np.pi*fin(phi), 0., np.pi, -np.pi/2., np.pi/2)
    c2 = 2/np.pi
    sph_heat_kernel_alt = ss[0]*c2

    sph_sim_geo = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.geodesic_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(jnp.arccos(x[0]/sph.r)),
        (sk, t_final, n_path, n_div, d0, d+1))

    sph_sim_geo_norm = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.geodesic_move_normalized(sph, x, unit_move, scale*d),
        None,
        lambda x: fin(jnp.arccos(x[0]/sph.r)),
        (sk, t_final, n_path, n_div, d0, d+1))

    sph_sim_ito = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.rbrownian_ito_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(jnp.arccos(x[0]/sph.r)),
        (sk, t_final, n_path, n_div, d0, d+1))

    sph_sim_strato = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.rbrownian_stratonovich_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(jnp.arccos(x[0]/sph.r)),
        (sk, t_final, n_path, n_div, d0, d+1))

    print(f"heat_kernels={sph_heat_kernel}")
    print(f"heat_kernels_alt={sph_heat_kernel_alt}")

    print(f"geodesic={jnp.mean(sph_sim_geo[0])}")
    print(f"geodesic={jnp.mean(sph_sim_geo_norm[0])}")
    print(f"ito={jnp.mean(sph_sim_ito[0])}")
    print(f"strato={jnp.mean(sph_sim_strato[0])}")


def test_3d():
    key = random.PRNGKey(0)
    key, sk = random.split(key)

    n = 4

    r = 3
    t_final = 2.

    d0 = .4
    n_path = 1000
    n_div = 1000
    sph = Sphere(n, r)

    def fin(phi):
        # return phi**2.5
        return phi**1.5 + phi**2.5
    # return phi**1.5 + phi**2.5 + phi**2.5
    
    sph_heat_kernel = jsi.trapezoid(
        np.array([hkm.k3(phi, t_final*d0/r**2)*(np.sin(phi))**2*4*np.pi*fin(phi)
                  for phi in np.arange(n_path+1)/n_path*np.pi]),
        dx=np.pi/n_path)

    sph_sim_geo = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.geodesic_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(jnp.arccos(x[0]/sph.r)),
        (sk, t_final, n_path, n_div, d0, n))

    sph_sim_geo_norm = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.geodesic_move_normalized(sph, x, unit_move, scale*(n-1)),
        None,
        lambda x: fin(jnp.arccos(x[0]/sph.r)),
        (sk, t_final, n_path, n_div, d0, n))
    print(jnp.mean(sph_sim_geo_norm[0]))

    sph_sim_ito = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.rbrownian_ito_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(jnp.arccos(x[0]/sph.r)),
        (sk, t_final, n_path, n_div, d0, n))

    sph_sim_strato = sim.simulate(
        sph.x0,
        lambda x, unit_move, scale: mi.rbrownian_stratonovich_move(sph, x, unit_move, scale),
        None,
        lambda x: fin(jnp.arccos(x[0]/sph.r)),
        (sk, t_final, n_path, n_div, d0, n))

    print(f"heat_kernels={sph_heat_kernel}")

    print(f"geodesic={jnp.mean(sph_sim_geo[0])}")
    print(f"geodesic={jnp.mean(sph_sim_geo_norm[0])}")
    print(f"ito={jnp.mean(sph_sim_ito[0])}")
    print(f"strato={jnp.mean(sph_sim_strato[0])}")


if __name__ == '__main__':
    test_1d()
    test_2d()
    test_3d()
