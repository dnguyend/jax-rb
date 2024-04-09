import os
import jax
import itertools
import jax.numpy as jnp
from jax import random
import jax_rb.manifolds.sphere as spm
import jax_rb.simulation.simulator as sim
import jax_rb.simulation.global_manifold_integrator as gmi


def run_all_sphere(save_dir):
    """ run all scenarios for so and save to save_dir
    """

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    n_path = 1000
    n_divs = [200, 500, 700]

    t_finals = [0.5, 2., 3., 10.]
    # dimension of the embedded space
    n_list = [3, 5, 10]
    d_coeffs = [.4]
    n_keys = 3

    key = random.PRNGKey(0)
    pay_offs = [(None,
                 lambda x: jnp.arccos(x[0]/r)**2.5),
                (lambda x, t: t/r*jnp.maximum(x[0]-r*.5, 0),
                 lambda x: jnp.arccos(x[0]/r)**2.5)
                ]

    r = 3

    def one_run(mnf, integrator, pars):
        """
        """
        n, normalized, int_name, t_final, n_div, d_coeff, subkey, n_path, stor = pars
        if normalized:
            dimsc = mnf.dim
        else:
            dimsc = 1.
        rpar = sim.RunParams(
            jnp.zeros(n).at[0].set(mnf.r), subkey, t_final, n_path,
            n_div, d_coeff*dimsc,
            n, n, normalized, int_name)
        # print(normalized, d_coeff, dimsc)
        stor.run(
            lambda x, unit_move, scale: integrator(
                mnf, x, unit_move, scale), rpar)

    for n in n_list:
        mnf = spm.Sphere(n, r)
        po_idx = 0
        for (path_pay_off, final_pay_off) in pay_offs:
            stor = sim.Simulator(path_pay_off, final_pay_off)
            print(f"doing n={n} pay_off_idx={po_idx}")
            for a in itertools.product(t_finals, n_divs, d_coeffs, range(n_keys)):
                key, subkey = random.split(key)
                t_final, n_div, d_coeff, _ = a
                one_run(mnf, gmi.geodesic_move,
                        (n, False, 'geodesic_move', t_final, n_div,
                         d_coeff, subkey, n_path, stor))
                one_run(mnf, gmi.geodesic_move_normalized,
                        (n, True, 'geodesic_move_normalized', t_final, n_div,
                         d_coeff, subkey, n_path, stor))
                one_run(mnf, gmi.rbrownian_ito_move,
                        (n, False, 'ito_move', t_final, n_div,
                         d_coeff, subkey, n_path, stor))
                one_run(mnf, gmi.rbrownian_stratonovich_move,
                        (n, False, 'stratonovich_move', t_final, n_div,
                         d_coeff, subkey, n_path, stor))

            stor.save_runs(os.path.join(save_dir, f"sphere_paper_{n}_{po_idx}"))
            po_idx += 1


if __name__ == '__main__':
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    import sys
    if len(sys.argv) < 2:
        print(f"Please run with format python {sys.argv[0]} [output_dir]. Files will be saved in [output_dir]/sphere")
        sys.exit()
    
    sdir = f"{sys.argv[1]}/sphere"
    print(sdir)
    # sdir = 'tests/output/sphere'
    run_all_sphere(sdir)
