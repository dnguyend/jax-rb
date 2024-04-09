import os
import itertools

import jax
import jax.numpy as jnp
from jax import random
import jax_rb.manifolds.spd as spdm

import jax_rb.simulation.simulator as sim
import jax_rb.simulation.global_manifold_integrator as gmi


def stratonovich_move_alt(mnf, x, unit_move, scale):
    """ Stratonovich version.
    Use Euler Heun
    """
    # stochastic dx
    smove = unit_move.reshape(mnf.shape)*jnp.sqrt(scale)
    dxs = mnf.sigma(x, smove)
    xbk = x + dxs
    return x + mnf.sigma(0.5*(x + xbk), smove) + mnf.stratonovich_drift(x)*scale


def run_all_spd(save_dir):
    """ run all scenarios for so and save to save_dir
    """
    # save_dir = '/home/dnguyen/dev/GeometricML/rbrown/runs/output/spd'
    # jax.config.update('jax_platform_name', 'cpu')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    n_path = 1000
    # ticks per second
    n_divs = [200, 500, 700]

    t_finals = [0.5, 2., 3., 10.]
    # dimension of the embedded space
    n_list = [3, 5]
    d_coeffs = [.5]
    n_keys = 3

    # big caveat: need slow function

    key = random.PRNGKey(0)
    pay_offs = [(None,
                 lambda x: (1+jnp.min(jnp.abs(x)))**(.5)),
                (lambda x, t: t*jnp.maximum(x[0, 0], 0),
                 lambda x: (1+jnp.min(jnp.abs(x)))**(.5))
                ]

    debug = False
    if debug:
        n_path = 1000
        n_divs = [10000]

        t_finals = [10.]
        # dimension of the embedded space
        n_list = [3]
        d_coeffs = [.5]
        n_keys = 1

        key = random.PRNGKey(0)
        pay_offs = [(None,
                     lambda x: jnp.abs(x[0, 0]))]

        pay_offs = [(lambda x, t: t*jnp.maximum(x[0, 0], 0),
                     lambda x: jnp.abs(x[0, 0]))]
        
        pay_offs = [(lambda x, t: t*jnp.maximum(x[0, 0], 0),
                     lambda x: (1+jnp.min(jnp.abs(x)))**(.5))]
        

    def one_run(mnf, integrator, pars):
        """One simulation
        """
        normalized, int_name, t_final, n_div, d_coeff, subkey, n_path, stor = pars
        if normalized:
            dimsc = mnf.dim
        else:
            dimsc = 1.
        n = mnf.shape[0]
        rpar = sim.RunParams(
            jnp.zeros((n, n)).at[:n, :n].set(jnp.eye(n)),
            subkey, t_final, n_path,
            n_div, d_coeff*dimsc,
            n*n, n, normalized, int_name)
        # print(normalized, d_coeff, dimsc)
        stor.run(
            lambda x, unit_move, scale: integrator(
                mnf, x, unit_move, scale), rpar)

    for n in n_list:
        mnf = spdm.PositiveDefiniteManifold(n)
        po_idx = 0
        for (path_pay_off, final_pay_off) in pay_offs:
            stor = sim.Simulator(path_pay_off, final_pay_off)
            print(f"doing mnf={mnf.name()} pay_off_idx={po_idx}")
            for a in itertools.product(t_finals, n_divs, d_coeffs, range(n_keys)):
                key, subkey = random.split(key)
                t_final, n_div, d_coeff, _ = a
                
                """
                one_run(mnf, gmi.geodesic_move_exact,
                        (False, 'geodesic_move_exact', t_final, n_div,
                         d_coeff, subkey, n_path, stor))
                one_run(mnf, gmi.geodesic_move_exact_normalized,
                        (True, 'geodesic_move_exact_normalized', t_final, n_div,
                         d_coeff, subkey, n_path, stor))
                """
                one_run(mnf, gmi.geodesic_move,
                        (False, 'geodesic_move', t_final, n_div,
                         d_coeff, subkey, n_path, stor))

                one_run(mnf, gmi.geodesic_move_normalized,
                        (True, 'geodesic_move_normalized', t_final, n_div,                         
                         d_coeff, subkey, n_path, stor))
                one_run(mnf, gmi.rbrownian_ito_move,
                        (False, 'ito_move', t_final, n_div,
                         d_coeff, subkey, n_path, stor))
                
                one_run(mnf, gmi.rbrownian_stratonovich_move,
                        (False, 'stratonovich_move', t_final, n_div,
                         d_coeff, subkey, n_path, stor))

                one_run(mnf, stratonovich_move_alt,
                        (False, 'stratonovich_move_alt', t_final, n_div,
                         d_coeff, subkey, n_path, stor))
                
            # stor.save_runs(os.path.join(save_dir, f"spd_paper_{n}_{po_idx}"))
            stor.save_runs(os.path.join(save_dir, f"spd_rerun_{n}_{po_idx}"))
            po_idx += 1
    # display([(jnp.nanmean(a[1]), a[0].run_type) for a in stor.runs])


if __name__ == '__main__':
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    import sys
    if len(sys.argv) < 2:
        print(f"Please run with format python {sys.argv[0]} [output_dir]. Files will be saved in [output_dir]/spd")
        sys.exit()

    sdir = f"{sys.argv[1]}/spd"
    print(sdir)

    run_all_spd(sdir)
