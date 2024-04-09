import os
import jax
import itertools
import jax.numpy as jnp
from jax import random
import jax_rb.manifolds.affine_left_invariant as afm
from jax_rb.utils.utils import (rand_positive_definite)
import jax_rb.simulation.simulator as sim
import jax_rb.simulation.matrix_group_integrator as mi


def run_all_affine(save_dir):
    """ run all scenarios for so and save to save_dir
    """
    # save_dir = '/home/dnguyen/dev/GeometricML/rbrown/runs/output/affine'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    n_path = 1000
    n_divs = [200, 500, 700]
    # n_divs = [200]

    t_finals = [.5, 2., 3., 10]
    # t_finals = [.3]
    # size of the group
    # n_list = [3, 4]
    # n_list = [1, 2]
    n_list = [1, 2, 3]
    d_coeffs = [.5]
    n_keys = 3

    pay_offs = [(None,
                 lambda x: (1+jnp.abs(x[0, 0]))**(-.5)),
                (lambda x, t: t*jnp.maximum(x[0, 0]-.5, 0),
                 lambda x: (1+jnp.abs(x[0, 0]))**(-.5))
                ]
    key = random.PRNGKey(0)

    debug = False
    if debug:
        n_divs = [200]
        # n_divs = [200]

        t_finals = [.5]
        # t_finals = [.3]
        # size of the group
        # n_list = [3, 4]
        n_list = [2]
        d_coeffs = [.5]
        n_keys = 1

        pay_offs = [(lambda x, t: t*jnp.maximum(x[0, 0]-.5, 0),
                     lambda x: (1+jnp.abs(x[0, 0]))**(-.5))
                    ]
        pay_offs = [(None,
                     lambda x: (1+jnp.abs(x[0, 0]))**(-.5))
                     # lambda x: jnp.abs(x[0, 0]))
                    ]


    # (lambda x, t: t*jnp.maximum(x[0, 0]-.5, 0),
    # lambda x: x[0, 0]**2)

    # pay_offs = [(None, lambda x: jnp.abs(x[0, 0]**2))]

    def one_run(mnf, integrator, pars):
        w_dim, normalized, int_name, t_final, n_div, d_coeff, subkey, n_path, stor = pars
        # n, size, normalized, int_name
        n1 = mnf.shape[0]
        if normalized:
            dimsc = mnf.dim
        else:
            dimsc = 1.
        rpar = sim.RunParams(
            jnp.eye(n1), subkey, t_final, n_path,
            n_div, d_coeff*dimsc,
            w_dim, mnf.name(), normalized, int_name)
        stor.run(
            lambda x, unit_move, scale: integrator(
                mnf, x, unit_move, scale), rpar)

    for n in n_list:
        aff_dim = n*(n+1)
        n1 = n + 1

        metric_mat, key = rand_positive_definite(key, aff_dim, (.1, 10.))
        mnf = afm.AffineLeftInvariant(n, metric_mat)
        # mnf = afm.AffineLeftInvariant(n, jnp.eye(aff_dim))
        po_idx = 0
        for (path_pay_off, final_pay_off) in pay_offs:
            stor = sim.Simulator(path_pay_off, final_pay_off)
            print(f"doing n={n} pay_off_idx={po_idx}")
            for a in itertools.product(t_finals, n_divs, d_coeffs, range(n_keys)):
                t_final, n_div, d_coeff, _ = a
                key, subkey = random.split(key)
                one_run(mnf, mi.geodesic_move,
                        (n1**2, False, 'geodesic_move',
                         t_final, n_div, d_coeff, subkey, n_path, stor))
                one_run(mnf, mi.geodesic_move_normalized,
                        (n1**2, True, 'geodesic_move_normalized',
                         t_final, n_div, d_coeff, subkey, n_path, stor))

                one_run(mnf, mi.rbrownian_ito_move,
                        (n1**2, False, 'ito_move',
                         t_final, n_div, d_coeff, subkey, n_path, stor))

                one_run(mnf, mi.rbrownian_stratonovich_move,
                        (n1**2, False, 'stratonovich_move',
                         t_final, n_div, d_coeff, subkey, n_path, stor))

            stor.save_runs(os.path.join(save_dir, f"affine_paper_{n}_{po_idx}"))
            po_idx += 1


if __name__ == '__main__':
    import sys
    
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    if len(sys.argv) < 2:
        print(f"Please run with format python {sys.argv[0]} [output_dir]. Files will be saved in [output_dir]/affine")
        sys.exit()

    sdir = f"{sys.argv[1]}/affine"
    print(sdir)

    run_all_affine(sdir)
