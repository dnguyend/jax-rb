import os
import csv

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random
import jax_rb.manifolds.so_left_invariant as som
import jax_rb.manifolds.stiefel as stm
import jax_rb.manifolds.grassmann as grm

from jax_rb.utils.utils import (rand_positive_definite, grand)
import jax_rb.simulation.simulator as sim
import jax_rb.simulation.matrix_group_integrator as mi
import jax_rb.simulation.global_manifold_integrator as gmi




run_dict = {'so': [som.SOLeftInvariant,
                   [3, 4],
                   [lambda x: jnp.abs(x[0, 0]**2),
                    lambda x: jnp.abs(jla.inv(x)[0, 0]**2),
                    lambda x: jnp.sum(jnp.abs(x)),
                    lambda x: jnp.sum(jnp.abs(jla.inv(x))),
                    lambda x: jnp.exp(.5*jnp.sum(jnp.abs(x))),
                    lambda x: jnp.exp(.5*jnp.sum(jnp.abs(jla.inv(x)))),
                    lambda x: (1+(jnp.sum(jnp.abs(x))))**(-.5),
                    lambda x: (1+(jnp.sum(jnp.abs(jla.inv(x)))))**(-.5)]],
            'st': [stm.RealStiefelAlpha,
                   [((5, 1), (1., 1)),
                    ((5, 3), (1., .5)),
                    ((5, 3), (1., .8)),
                    ((5, 3), (1., 1))],
                   [lambda x: jnp.abs(x[0, 0]**2),
                    lambda x: jnp.sum(jnp.abs(x)),
                    lambda x: jnp.exp(0.5*jnp.sum(jnp.abs(x))),
                    lambda x: (1+(jnp.sum(jnp.abs(x))))**(-.5)]],
            'gr': [grm.Grassmann,
                   [(5, 3)],
                   [lambda x: jnp.abs((x@x.T)[0, 0]**2),
                    lambda x: jnp.sum(jnp.abs(x@x.T)),
                    lambda x: jnp.exp(0.5*jnp.sum(jnp.abs(x@x.T))),
                    lambda x: (1+(jnp.sum(jnp.abs(x@x.T))))**(-.5)]]}


def uniform_sample(key, shape, pay_off, n_samples):
    """ Sample the manifold uniformly
    """
    x_all, key = grand(key, (shape[0], shape[1], n_samples))

    def do_one_point(seq):
        ei, ev = jla.eigh(seq.T@seq)
        return pay_off(seq@ev@((1/jnp.sqrt(ei))[:, None]*ev.T))

    s = jax.vmap(do_one_point, in_axes=2)(x_all)
    return jnp.nanmean(s)


def man_dim(m_key, a_def):
    """ dimension of manifold
    """
    if m_key == 'so':
        n = a_def
        return n*(n-1)//2, n**2, jnp.eye(n)
    if m_key == 'st':
        n, p = a_def[0]
        return p*(p-1)//2 + (n-p)*p, n*p, jnp.zeros((n, p)).at[:p, :].set(jnp.eye(p))
    if m_key == 'gr':
        n, p = a_def
        return (n-p)*p, n*p, jnp.zeros((n, p)).at[:p, :].set(jnp.eye(p))
    return  None


def sample_all(n_path, t_finals):
    # first run the uniform samping
    key = random.PRNGKey(0)
    ret = {}
    for m_key, value in run_dict.items():
        _, m_shape, pay_offs = value
        if m_key == 'st':
            shapes = [a[0] for a in m_shape]
        elif m_key == 'so':
            shapes = [(a, a) for a in m_shape]
        else:
            shapes = m_shape

        for shape in shapes:
            for po_idx in range(len(pay_offs)):
                ret[m_key, shape, po_idx] = uniform_sample(key, shape,
                                                           pay_offs[po_idx],
                                                           int(t_finals[-1]*n_path))
    jnp.savez(os.path.join(SAVE_DIR, 'sampling'),
              {str(a): val for a, val in ret.items()}, allow_pickle=True)
    return ret


def sim_all(n_path, t_finals, h, d_coeff):
    """ run all scenarios for so and save to SAVE_DIR
    """
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    key = random.PRNGKey(0)


    def one_run(mnf, x_0, integrator, pars):
        w_dim, normalized, int_name, t_final, n_div, d_coeff, subkey, n_path, stor = pars
        # n, size, normalized, int_name
        if normalized:
            dimsc = mnf.dim
        else:
            dimsc = 1.
        rpar = sim.RunParams(
            x_0, subkey, t_final, n_path,
            n_div, d_coeff*dimsc,
            w_dim, mnf.name(), normalized, int_name)
        stor.run(
            lambda x, unit_move, scale: integrator(
                mnf, x, unit_move, scale), rpar)

    for m_key, value in run_dict.items():
        m_class, m_def, pay_offs = value
        for a_def in m_def:
            mnf_dim, ambient_dim, x0 = man_dim(m_key, a_def)
            if m_key == 'so':
                metric_mat, key = rand_positive_definite(key, mnf_dim, (.1, 10.))
                ei, ev = jla.eigh(metric_mat)
                mnf = m_class(a_def, metric_mat)
                # mnf = m_class(a_def, jnp.eye(a_def*(a_def-1)//2))
            elif m_key == 'st':
                mnf = m_class(a_def[0], jnp.array(a_def[1]))
            elif m_key == 'gr':
                mnf =  m_class(a_def)
            po_idx = 0
            for final_pay_off in pay_offs:
                stor = sim.Simulator(None, final_pay_off)
                print(f"doing {mnf.name()} pay_off_idx={po_idx}")
                for t_final in t_finals:
                    key, subkey = random.split(key)
                    n_div = int(t_final/h)
                    if m_key == 'so':
                        one_run(mnf, x0, mi.geodesic_move,
                                (ambient_dim, False, 'geodesic_move',
                                 t_final, n_div, d_coeff, subkey, n_path, stor))
                    else:
                        one_run(mnf, x0, gmi.geodesic_move,
                                (ambient_dim, False, 'geodesic_move',
                                 t_final, n_div, d_coeff, subkey, n_path, stor))
                    """
                    one_run(mnf, mi.geodesic_move_dim_g,
                            (mnf.dim, False, 'geodesic_move_dim_g',
                             t_final, n_div, d_coeff, subkey, n_path, stor))
                    """
                    # integrator = mi.geodesic_move_normalized if m_key == 'so' else gmi.geodesic_move_normalized
                    if m_key == 'so':
                        one_run(mnf, x0, mi.geodesic_move_normalized,
                                (ambient_dim, True, 'geodesic_move_normalized',
                                 t_final, n_div, d_coeff, subkey, n_path, stor))
                    else:
                        one_run(mnf, x0, gmi.geodesic_move_normalized,
                                (ambient_dim, True, 'geodesic_move_normalized',
                                 t_final, n_div, d_coeff, subkey, n_path, stor))
                    """
                    one_run(mnf, mi.geodesic_move_dim_g_normalized,
                            (mnf.dim, True, 'geodesic_move_dim_g_normalized',
                             t_final, n_div, d_coeff, subkey, n_path, stor))
                    """
                    # integrator = mi.rbrownian_ito_move if m_key == 'so' else gmi.rbrownian_ito_move
                    if m_key == 'so':
                        one_run(mnf, x0, mi.rbrownian_ito_move,
                                (ambient_dim, False, 'ito_move',
                                 t_final, n_div, d_coeff, subkey, n_path, stor))
                    else:
                        one_run(mnf, x0, gmi.rbrownian_ito_move,
                                (ambient_dim, False, 'ito_move',
                                 t_final, n_div, d_coeff, subkey, n_path, stor))
                    """
                    one_run(mnf, mi.ito_move_dim_g,
                            (mnf.dim, False, 'ito_move_dim_g',
                             t_final, n_div, d_coeff, subkey, n_path, stor))
                    """
                    # integrator = mi.rbrownian_rbrownian_move if m_key == 'so' else gmi.rbrownian_stratonovich_move
                    if m_key == 'so':
                        one_run(mnf, x0, mi.rbrownian_stratonovich_move,
                                (ambient_dim, False, 'stratonovich_move',
                                 t_final, n_div, d_coeff, subkey, n_path, stor))
                    else:
                        one_run(mnf, x0, gmi.rbrownian_stratonovich_move,
                                (ambient_dim, False, 'stratonovich_move',
                                 t_final, n_div, d_coeff, subkey, n_path, stor))
                    """
                    one_run(mnf, mi.stratonovich_move_dim_g,
                            (mnf.dim, False, 'stratonovich_move_dim_g',
                             t_final, n_div, d_coeff, subkey, n_path, stor))
                    """
                stor.save_runs(os.path.join(SAVE_DIR, f"{mnf.name()}_{po_idx}"))
                print([(jnp.nanmean(a[1]), a[0].run_type, po_idx) for a in stor.runs])
                po_idx += 1


def parse_fname(fname):
    if fname.lower().startswith('gr'):
        n, p = [int(a) for a in fname[3:-7].split(', ')]
        po_idx = int(fname[-5:-4])
        return ('gr', (n, p), po_idx), '[]'
    if fname.lower().startswith('stiefel'):
        # n, p = [int(a) for a in fname[9:-7].split(', ')]
        sstr, alstr = fname[9:-7].split(')) alpha=[')
        n, p = [int(a) for a in sstr.split(',')]
        al = [float(a) for a in alstr.split()]
        po_idx = int(fname[-5:-4])
        return ('st', (n, p), po_idx), al
    if fname.lower().startswith('so'):
        n = int(fname[3:4])
        po_idx = int(fname[-5:-4])
        return ('so', (n, n), po_idx), '[]'


def analyze_files_0(ret):
    """ Analyze the run
    """
    fnames = sorted([a for a in os.listdir(SAVE_DIR)
                     if (a.endswith('.npz') and (a != 'sampling.npz'))])

    rpt = {}
    for fname in fnames:
        ret_key, al = parse_fname(fname)
        mtype, shape, po_idx = ret_key
        f1 = jnp.load(os.path.join(SAVE_DIR, fname), allow_pickle=True)
        fsims = []
        for i in range(f1['pay_offs'].shape[0]):
            if f1['params'][i]['t_final'] >=39.0:
                fsims.append(jnp.nanmean(f1['pay_offs'][i, :]))
        fsims = jnp.array(fsims)
        # print(fname, ret_key, ret[ret_key], al, jnp.max(fsims),jnp.min(fsims), jnp.mean(fsims))
        new_key = (mtype, shape, str(al))
        if new_key not in rpt.keys():
            if po_idx < 4:
                rpt[new_key] = {
                    po_idx: {"sample": ret[ret_key],
                             "max_sim": jnp.max(fsims),
                             "min_sim": jnp.min(fsims),
                             "mean_sim": jnp.mean(fsims)
                             }}
            else:
                rpt[new_key] = {
                    po_idx-4: {'sample': ret[ret_key],
                               "inv_max_sim": jnp.max(fsims),
                               "inv_min_sim": jnp.min(fsims),
                               "inv_mean_sim": jnp.mean(fsims)
                               }}
        else:
            if po_idx < 4:
                if po_idx not in rpt[new_key].keys():
                    rpt[new_key][po_idx] = {"sample":  ret[ret_key],
                                            "max_sim": jnp.max(fsims),
                                            "min_sim": jnp.min(fsims),
                                            "mean_sim": jnp.mean(fsims)}
                else:
                    rpt[new_key][po_idx]["sample"] =  ret[ret_key]
                    rpt[new_key][po_idx]["max_sim"] = jnp.max(fsims)
                    rpt[new_key][po_idx]["min_sim"] = jnp.min(fsims)
                    rpt[new_key][po_idx]["mean_sim"] = jnp.mean(fsims)
            else:
                if po_idx-4 not in rpt[new_key].keys():
                    rpt[new_key][po_idx-4] = {"inv_sample":  ret[ret_key],
                                              "inv_max_sim": jnp.max(fsims),
                                              "inv_min_sim": jnp.min(fsims),
                                              "inv_mean_sim": jnp.mean(fsims)}
                else:
                    rpt[new_key][po_idx-4]["inv_sample"] =  ret[ret_key]
                    rpt[new_key][po_idx-4]["inv_max_sim"] = jnp.max(fsims)
                    rpt[new_key][po_idx-4]["inv_min_sim"] = jnp.min(fsims)
                    rpt[new_key][po_idx-4]["inv_mean_sim"] = jnp.mean(fsims)


def analyze_files(ret):
    """ Analyze the run
    """
    fnames = sorted([a for a in os.listdir(SAVE_DIR)
                     if (a.endswith('.npz') and (a != 'sampling.npz'))])

    rpt = {}
    for fname in fnames:
        ret_key, al = parse_fname(fname)
        mtype, shape, po_idx = ret_key
        f1 = jnp.load(os.path.join(SAVE_DIR, fname), allow_pickle=True)
        fsims = []
        for i in range(f1['pay_offs'].shape[0]):
            if f1['params'][i]['t_final'] >=39.0:
                fsims.append(jnp.nanmean(f1['pay_offs'][i, :]))
        fsims = jnp.array(fsims)
        # print(fname, ret_key, ret[ret_key], al, jnp.max(fsims),jnp.min(fsims), jnp.mean(fsims))
        new_key = (mtype, shape, str(al))
        if mtype != 'so':
            if new_key not in rpt:
                rpt[new_key] = {
                    po_idx: {"sample": ret[ret_key],
                             "sim_rng": jnp.max(fsims)-jnp.min(fsims),
                             "mean_sim": jnp.mean(fsims)
                             }}
            else:
                if po_idx not in rpt[new_key].keys():
                    rpt[new_key][po_idx] = {"sample":  ret[ret_key],
                                            "sim_rng": jnp.max(fsims)-jnp.min(fsims),
                                            "mean_sim": jnp.mean(fsims)}
                else:
                    rpt[new_key][po_idx]["sample"] =  ret[ret_key]
                    rpt[new_key][po_idx]["sim_rng"] = jnp.max(fsims)-jnp.min(fsims)
                    rpt[new_key][po_idx]["mean_sim"] = jnp.mean(fsims)
        else:
            if new_key not in rpt:
                if po_idx % 2 == 0:
                    rpt[new_key] = {
                        po_idx//2: {"sample": ret[ret_key],
                                 "sim_rng": jnp.max(fsims)-jnp.min(fsims),
                                 "mean_sim": jnp.mean(fsims)
                                 }}
                else:
                    rpt[new_key] = {
                        (po_idx-1)//2: {'sample': ret[ret_key],
                                   "inv_sim_rng": jnp.max(fsims) - jnp.min(fsims),
                                   "inv_mean_sim": jnp.mean(fsims)
                                   }}
            else:
                if po_idx % 2 == 0:
                    po_hlf = po_idx // 2
                    if po_hlf not in rpt[new_key].keys():
                        rpt[new_key][po_hlf] = {"sample":  ret[ret_key],
                                                "sim_rng": jnp.max(fsims)-jnp.min(fsims),
                                                "mean_sim": jnp.mean(fsims)}
                    else:
                        rpt[new_key][po_hlf]["sample"] =  ret[ret_key]
                        rpt[new_key][po_hlf]["sim_rng"] = jnp.max(fsims)-jnp.min(fsims)
                        rpt[new_key][po_hlf]["mean_sim"] = jnp.mean(fsims)
                else:
                    po_hlf = (po_idx-1)//2
                    if po_hlf not in rpt[new_key].keys():
                        rpt[new_key][po_hlf] = {"inv_sample":  ret[ret_key],
                                                "inv_sim_rng": jnp.max(fsims)-jnp.min(fsims),
                                                "inv_mean_sim": jnp.mean(fsims)}
                    else:
                        rpt[new_key][po_hlf]["inv_sample"] =  ret[ret_key]
                        rpt[new_key][po_hlf]["inv_sim_rng"] = jnp.max(fsims)-jnp.min(fsims)
                        rpt[new_key][po_hlf]["inv_mean_sim"] = jnp.mean(fsims)

    import numpy as np

    def make_name(mtype, shape, alpha):
        if mtype == 'so':
            return f"SO({shape[0]})"
        if mtype == 'gr':
            return f"Gr_{{{shape[0],shape[1]}}}"
        if (mtype == 'st'):
            if shape[1] == 1:
                return f"S^{{{shape[0]}}}"
            else:
                return f"St_{{{shape[0]},{shape[1]}) ({alpha})}}"
        return None
    
    r = len(rpt.keys())
    c = 9
    tbl = np.empty((r, c), dtype=object)
    i = 0
    # idx = []
    for k, v in sorted(rpt.items()):
        mtype, shape, alpha = k
        if mtype == 'so':
            for po_idx in range(4):
                tbl[i, po_idx*2+1] = f"{v[po_idx]['sample']: .3f}|{v[po_idx]['inv_sample']: .3f}"
                tbl[i, po_idx*2+2] = f"{v[po_idx]['mean_sim']: .3f}({v[po_idx]['sim_rng']:.3f})|{v[po_idx]['inv_mean_sim']: .3f}({v[po_idx]['inv_sim_rng']:.3f})"
        else:
            for po_idx in range(4):            
                tbl[i, po_idx*2+1] = f"{v[po_idx]['sample']: .3f}"
                tbl[i, po_idx*2+2] = f"{v[po_idx]['mean_sim']: .3f}({v[po_idx]['sim_rng']:.4f})"
        tbl[i, 0] = make_name(mtype, shape, alpha)
        # idx.append(k)
        i += 1

    print(tbl)
    outfile = os.path.join(SAVE_DIR, 'compact_compare_uniform.csv')
    header = ['f', '1', '1', '2', '2', '3', '3', '4', '4']
    second_header = ['Manifold'] + 4*['sample', 'sim']
    with open(outfile, 'w', encoding="utf-8") as csvfile:
        cwrite = csv.writer(csvfile, delimiter=',')
        cwrite.writerow(header)
        cwrite.writerow(second_header)
        for row in tbl:
            cwrite.writerow(row)
    

if __name__ == '__main__':
    import sys
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    jax.config.update("jax_enable_x64", True)
    N_PATH = 1000
    if len(sys.argv) < 2:
        print("Please run with format python run_uniform.py [output_dir]. Files will be saved in [output_dir]/uniform")
        sys.exit()

    SAVE_DIR = f"{sys.argv[1]}/uniform"

    # n_divs = [200]

    T_FINALS = [3., 20, 40.]
    # t_finals = [40.]
    H = .01
    D_COEFF = .5
    sim_all(N_PATH, T_FINALS, H, D_COEFF)
    samples = sample_all(N_PATH, T_FINALS)
    analyze_files(samples)
