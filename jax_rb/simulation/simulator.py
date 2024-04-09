"""Simulator for global_manifold
"""
from collections import namedtuple

import jax.numpy as jnp
# import jax.numpy.linalg as jla
from jax import random, vmap


class RunParams(namedtuple('RunParams',
                           ['x_0', 'key', 't_final', 'n_path',
                            'n_div', 'd_coeff',
                            'wiener_dim', 'm_size', 'normalize', 'run_type'])):
    """Parameters to save a run in simulator.

    :param x_0: starting point of the simulation
    :param key: key to generate the random numbers used in simulation. Created from jax.random.PRNGKey, then jax.random.split.
    :param t_final: The final time of simulation. Starting time is :math:`t=0`.
    :param n_path: number of paths used in simulation
    :param n_div: number of subdivision (interval will be t_final/n_div
    :param d_coeff: difusion coefficient, d_coeff = 0.5 for the Riemannian Brownian motion.
    :param wiener_dim: dimension of the Wienner process used in simulation. Usually it is the dimension of the ambient space :math:`\\mathcal{E}`. In some cases, we can simulate using the dimension of the manifold itself.
    :param m_size: a param indicating the size of the manifold, use to differentiate when simulating several manifolds,
    :param normalize: whether to normalize the move to a fixed lengh,
    :param run_type: string indicating one of the simulation moves. This is a tag to distinguish the output, does not affect the results.
    """


def simulate(x_0,
             integrator,
             path_pay_off,
             final_pay_off,
             params):
    """A simulation from :math:`t=0` up to time :math:`t=t_final`, with time increment
    :math:`t=\\frac{t_final}{n_div}`, run :math:`n_path` path.
    return the full distribution of the simulation.

    :param x_0: starting point of the simulation
    :param integrator: one of the integrators (geodesic, ito, stratonovich
    :param path_pay_off: the cost evaluated along the path
    :param final_pay_off: the contribution evaluated at the final time
    :param params: additional parameters for the simulations: sk, t_final, n_path, n_div, d_coeff, wiener_dim
    """
    sk, t_final, n_path, n_div, d_coeff, wiener_dim = params
    x_all = random.normal(sk, (wiener_dim, n_div, n_path))

    def do_one_path(seq):
        path_sum = 0.
        x_i = x_0.copy()
        for j in range(n_div):
            x_i = integrator(x_i, seq[:, j], t_final/n_div*2*d_coeff)

            if path_pay_off:
                path_sum += path_pay_off(x_i, j*t_final/n_div)*t_final/n_div

        return path_sum + final_pay_off(x_i)

    # batch_do_one_path = jax.vmap(do_one_path, in_axes=2)
    pay_offs = vmap(do_one_path, in_axes=2)(x_all)

    return pay_offs, x_all


class Simulator():
    """ Class to do simulation on a manifold
    with particular funtion or simulators. Run results is saved in self.runs.

    :param path_pay_off is the function value evaluated along the path
    :param final_pay_off is the function value evaluated at final time
    """
    def __init__(self, path_pay_off,
                 final_pay_off):
        self.path_pay_off = path_pay_off
        self.final_pay_off = final_pay_off
        self.runs = []

    def run(self, integrator, params):
        """ run a simulation
        
        :param integrator the integrator used
        :param params is of class RunParams
        """
        sim_params = (params.key, params.t_final, params.n_path,
                      params.n_div, params.d_coeff,
                      params.wiener_dim)
        pay_offs, _ = simulate(params.x_0, integrator, self.path_pay_off,
                               self.final_pay_off, sim_params)
        self.runs.append([params, pay_offs])

    def save_runs(self, save_path):
        """ save all the runs to save_path
        """
        idx = 0
        pay_offs = []
        prms = []
        while idx < len(self.runs):
            pay_offs.append(self.runs[idx][1])
            fi = self.runs[idx][0]._fields
            pp = {}
            for i in range(len(fi)):
                if fi[i] != 'x_0':
                    pp[fi[i]] = self.runs[idx][0][i]
            prms.append(pp)
            idx += 1
        jnp.savez(save_path, pay_offs=pay_offs,
                  params=prms, allow_pickle=True)
