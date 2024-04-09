import os
import jax
import itertools
import jax.numpy as jnp
from jax import random
import jax_rb.manifolds.stiefel as stm

import jax_rb.simulation.simulator as sim
import jax_rb.simulation.global_manifold_integrator as gmi


def run_all_stiefel(save_dir):
    """ run all scenarios for so and save to save_dir
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    n_path = 1000
    n_divs = [200, 500, 700]

    t_finals = [0.5, 2., 3., 10.]
    # dimension of the embedded space
    n_list = [((5, 3), (1., .5)),
              ((5, 3), (1., 1.)),
              ((5, 3), (1., .8)),
              ((7, 4), (1., .5)),
              ((7, 4), (1., 1.))
              ]
    d_coeffs = [.5]
    n_keys = 3

    key = random.PRNGKey(0)
    pay_offs = [(None,
                 lambda x: jnp.abs(x[0, 0])),
                (lambda x, t: t*jnp.maximum(x[0, 0], 0),
                 lambda x: jnp.abs(x[0, 0]))
                ]

    debug = False
    if debug:
        n_divs = [200]

        t_finals = [.5]
        # dimension of the embedded space
        n_list = [((5, 3), jnp.array([1., .5]))]
        d_coeffs = [.5]
        n_keys = 1

        key = random.PRNGKey(0)
        pay_offs = [(None,
                     lambda x: jnp.abs(x[0, 0]))
                    ]
        pay_offs = [(lambda x, t: t*jnp.maximum(x[0, 0], 0),
                     lambda x: jnp.abs(x[0, 0]))]

    def one_run(mnf, integrator, pars):
        """One simulation
        """
        normalized, int_name, t_final, n_div, d_coeff, subkey, n_path, stor = pars
        if normalized:
            dimsc = mnf.dim
        else:
            dimsc = 1.
        n, p = mnf.shape
        alpha = mnf.alpha
        sh_al = f"{n}:{p}:{alpha[0]}:{alpha[1]}"
        rpar = sim.RunParams(
            jnp.zeros((n, p)).at[:p, :p].set(jnp.eye(p)),
            subkey, t_final, n_path,
            n_div, d_coeff*dimsc,
            n*p, sh_al, normalized, int_name)
        # print(normalized, d_coeff, dimsc)
        stor.run(
            lambda x, unit_move, scale: integrator(
                mnf, x, unit_move, scale), rpar)

    for shape, alpha in n_list:
        mnf = stm.RealStiefelAlpha(shape, jnp.array(alpha))
        po_idx = 0
        for (path_pay_off, final_pay_off) in pay_offs:
            stor = sim.Simulator(path_pay_off, final_pay_off)
            print(f"doing mnf={mnf.name()} pay_off_idx={po_idx}")
            for a in itertools.product(t_finals, n_divs, d_coeffs, range(n_keys)):
                key, subkey = random.split(key)
                t_final, n_div, d_coeff, _ = a
                one_run(mnf, gmi.geodesic_move_exact,
                        (False, 'geodesic_move_exact', t_final, n_div,
                         d_coeff, subkey, n_path, stor))
                one_run(mnf, gmi.geodesic_move_exact_normalized,
                        (True, 'geodesic_move_exact_normalized', t_final, n_div,
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
            stor.save_runs(os.path.join(save_dir, f"stiefel_paper_{shape[0]}:{shape[1]}:{alpha[0]}:{alpha[1]}_{po_idx}"))
            po_idx += 1
                

if __name__ == '__main__':
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    import sys
    if len(sys.argv) < 2:
        print(f"Please run with format python {sys.argv[0]} [output_dir]. Files will be saved in [output_dir]/stiefel")
        sys.exit()

    sdir = f"{sys.argv[1]}/stiefel"
    print(sdir)

    run_all_stiefel(sdir)
