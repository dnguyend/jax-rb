"""collect results from the output/[manifold] dir and save in summaries
"""
import os
import csv
import numpy as np
import pandas as pd


def grassmann_dim(s):
    """ parse the string to get the dimension of the grassmann
    """
    n, p = [int(a) for a in s.split(':')[:2]]
    return (n-p)*p

def stiefel_dim(s):
    """ parse the string to get the dimension of the grassmann
    """
    n, p = [int(a) for a in s.split(':')[:2]]
    return p*(p-1)//2 + (n-p)*p


dirs = {'so': {'func': lambda s: (int(s)**2-int(s))//2,
               'picked': ['so_paper_3_0.npz', 'so_paper_3_1.npz'],
               'short': 'SO',
               'mtype': 'group'
               },
        'se': {'func': lambda s: (int(s)**2+int(s))//2,
               'picked': ['se_paper_3_0.npz', 'se_paper_3_1.npz'],
               'short': 'SE',
               'mtype': 'group'                   
               },
        'sl': {'func': lambda s: int(s)**2-1,
               'picked': ['sl_paper_3_0.npz', 'sl_paper_3_1.npz'],
               'short': 'SL',
               'mtype': 'group'                   
               },
        'affine': {'func': lambda s: int(s)**2+int(s),
                   'picked': ['affine_paper_3_0.npz', 'affine_paper_3_1.npz'],
                   'short': 'Aff',
                   'mtype': 'group'
                   },
        'glp': {'func': lambda s: int(s)**2,
                'picked': ['glp_paper_2_0.npz', 'glp_paper_2_1.npz'],
                'short': 'GL^+',
                'mtype': 'group'                    
                },
        'sphere': {'func': lambda s: int(s)-1,
                   'picked': ['sphere_paper_10_0.npz', 'sphere_paper_10_1.npz'],
                   'short': 'S',
                   'mtype': 'manifold'
                   },
        'grassmann': {'func': grassmann_dim,
                      'picked': ['grassmann_paper_5:3_0.npz', 'grassmann_paper_5:3_1.npz'],
                      'short': 'Gr',
                      'mtype': 'manifold'
                      },
        'spd': {'func': lambda s: (int(s)**2+int(s))//2,
                'picked': ['spd_paper_3_0.npz', 'spd_paper_3_1.npz'],
                'short': 'S^+',
                'mtype': 'manifold'                    
                },
        'stiefel': {'func': stiefel_dim,
                    'picked': ['stiefel_paper_5:3:1.0:0.5_0.npz',
                               'stiefel_paper_5:3:1.0:0.5_1.npz'
                               ],
                    'short': 'St',
                    'mtype': 'manifold'
                    }}


header = ['manifold', 'manifold type', 'shape', 'fname', 'cost_type',
          'expected value',
          'key', 't_final', 'n_path',
          'n_div', 'd_coeff', 'w_dim', 'manifold(dim)',
          'normalized', 'run_type']


def cost_type(f):
    """ parse the cost type (0 or 1) from the name
    """
    return int(f[:-4].split('_')[-1])


def shape_type(d, f):
    "parse name for the shape "
    if d in ['so', 'se', 'sl', 'sphere', 'glp', 'spd']:
        return f.split('_')[2]
    return ", ".join(f.split('_')[2].split(':')[:2])


def run_type_map(a):
  if 'geodesic' in a.lower():
    return 'geodesic'
  if 'stratonovich' in a.lower():
    return 'stratonovich'
  if 'ito_' in a.lower():
    return 'ito'
  return None


def process_file_list():
    """Collect all files
    and produce a number of summaries
    """
    for d, ddir in dirs.items():
        print(f"dir={d}")
        curr_dir = os.path.join(base_dir, d)
        files = [a for a in os.listdir(curr_dir) if '_paper_' in a]
        # print(files, dirs[d]['picked'])
        # for f in dirs[d]['picked']:
        for f in files:
            print(f"{f} {ddir['short']} {cost_type(f)} {shape_type(d, f)}")

    # step one: select all files and analyze with panda
    all_rec = []
    for d, ddir in dirs.items():
        print(f"dir={d}")
        curr_dir = os.path.join(base_dir, d)
        files = [a for a in os.listdir(curr_dir) if (('_paper_' in a) and (a.endswith('.npz')))]
        # print(files, dirs[d]['picked'])
        # for f in dirs[d]['picked']:
        for f in files:
            for u in analyze_one(curr_dir, f, dirs[d]['func']):
                rec = [d, ddir['mtype'], shape_type(d, f), f, cost_type(f)]
                rec.extend(u)
                rec[-3] = f"{ddir['short']}({shape_type(d, f)})"
                all_rec.append(rec)

    outfile = os.path.join(base_dir, "all_sim_detail.csv")
    with open(outfile, 'w', encoding="utf-8") as csvfile:
        cwrite = csv.writer(csvfile, delimiter=',')
        cwrite.writerow(header)
        for a in all_rec:
            cwrite.writerow(a)

    return outfile


def analyze_one(fdir, fname, dim_formula):
    """ analyze one dir
    """
    f1 = np.load(os.path.join(fdir, fname), allow_pickle=True)

    dim = dim_formula(fname.split('_')[-2])

    def process_line(i):
        newline = [np.nanmean(f1["pay_offs"][i, :])]+list(f1['params'][i].values())
        if newline[-2]:
            newline[-5] /= dim
            # print(dim, newline[-5], newline)
        return newline
    ret = [process_line(i) for i in range(f1["pay_offs"].shape[0])]

    return sorted(ret, key=lambda x: (x[2], x[5], x[7], x[3], x[4], x[9],
                                          ))  # time, diffusion, manifolds


def create_latex_tables(detail_file):
    """ output a few latex tables
    """
    pbook = pd.read_csv(detail_file, header=0)

    flist = [a for d, ddir in dirs.items() for a in ddir['picked']]
    df2 = pd.DataFrame(pbook.loc[pbook.fname.isin(flist)][
        ['manifold type', 'manifold', 'fname', 'cost_type',
         't_final', 'n_div', 'run_type', 'expected value']
    ].pivot_table(
        values=['expected value'], index=['manifold type', 'manifold', 'n_div'],
        columns = ['cost_type', 't_final'],
        aggfunc=['mean', 'max', 'min']))

    agg_ord ={'mean': 0, 'max':2, 'min':1}

    new_cols = pd.MultiIndex.from_tuples(
        sorted(df2.columns, key=lambda a: (a[2], a[3], agg_ord[a[0]])))

    df3 = pd.DataFrame(df2, columns =new_cols)

    r, c = 27, 8
    table = np.empty((r, c), dtype=object)
    for i in range(c):
        for j in range(r):
            table[j, i] = f"{df3.iloc[j, 3*i].round(2)} ({(df3.iloc[j, 3*i+2]-df3.iloc[j, 3*i+1]).round(3)})"

    cost_dict ={0: 'At final time only', 1: 'With intermediate cost'}
    df4 = pd.DataFrame(table[:, :], index = pd.MultiIndex.from_tuples(
        [(a[1], a[2]) for a in df3.index], names=['manifold', '# div']),
                       columns= pd.MultiIndex.from_tuples(
                           [(cost_dict[df3.columns[i][2]], df3.columns[i][3])
                            for i in range(len(df3.columns)) if i %3==0],
                           names=['Cost type', 'Final time']))


    pbook.insert(2, 'short_type', pbook.run_type.map(run_type_map))

    df5 = pd.DataFrame(pbook.loc[np.logical_and(pbook.fname.isin(flist), pbook.n_div == 700)][
        ['manifold type', 'manifold', 'fname', 'cost_type', 't_final', 'n_div', 'short_type', 'expected value']
    ].pivot_table(
        values=['expected value'], index=['manifold type', 'manifold', 'short_type'],
        columns = ['cost_type', 't_final'],
    )).round(3)

    df5 = pd.DataFrame(df5.values, index=pd.MultiIndex.from_tuples([a[1:] for a in df5.index]),
                       columns= pd.MultiIndex.from_tuples([(cost_dict[df5.columns[i][1]], df5.columns[i][2]) for i in range(len(df5.columns))],
                                                           names=['cost type', 'final time']))

    df4.to_latex(latex_n_div)
    df5.to_latex(latex_sim_type)
    
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Please run with format python {sys.argv[0]} [output_dir]. Files will be saved in [output_dir]")
        sys.exit()

    # sdir = f"{sys.argv[1]}/uniform"
    # print(sdir)
    base_dir = sys.argv[1]
    out_file = process_file_list()

    latex_n_div = os.path.join(base_dir, "manifold_by_cost_type_time_division_final_time.tex")
    latex_sim_type = os.path.join(base_dir, "manifold_by_cost_type_sim_type_final_time.tex")
    create_latex_tables(out_file)
