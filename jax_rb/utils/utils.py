"""various utils for the project
"""
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random, jit


def pos(x):
    """ max(x, 0), jit friendly
    """
    return 0.5*(x+jnp.abs(x))


def asym(mat):
    return 0.5*(mat - mat.T)


def sym2(mat):
    return mat + mat.T


def sym(a):
    return 0.5*sym2(a)


def grand(key, dims):
    """ generate a random array of shape dim using key
    """
    key, sk = random.split(key)
    return random.normal(sk, dims), key


def lyapunov(a, b):
    """solve aU + Ua = b
    A, B, U are symmetric
    """
    yei, yv = jla.eigh(a)
    return yv@((yv.T@b@yv)/(yei[:, None] + yei[None, :]))@yv.T


def vcat(x, y):
    """vertical concatenate
    """
    return jnp.concatenate([x, y], axis=0)


def sinc(x):
    """ better sinc than
    """
    if jnp.abs(x) <= 1e-20:
        return 1
    return jnp.sin(x)/x


def sinc1(x):
    """sinc1 is dsinc/x
    """
    if jnp.abs(x) < 1e-6:
        return -1/3 + x*x/2/3/5
    return (x*jnp.cos(x)-jnp.sin(x))/x/x/x


def sinc2(x):
    """ helper  function, derivative of sinc1 = x*sinc2(x)
    """
    if jnp.abs(x) < 1e-3:
        return 1/15 - x*x/210 + x*x*x*x / 7560
    return -((x*x-3)*jnp.sin(x) + 3*x*jnp.cos(x))/x**5


def dsinc(x):
    """Derivative of sinc
    """
    if jnp.abs(x) < 1e-6:
        return -1/3*x + x*x*x/30
    return (x*jnp.cos(x)-jnp.sin(x))/x/x


def dsinc1(x):
    """sinc1 is dsinc/x
    dsinc1 is its derivative
    """
    return x*sinc2(x)

def complement_basis_for_vector(xraw):
    """ complement basis of xraw, a non zero vector. Assume  x[0] !=0
    """
    x = xraw/jnp.sqrt(jnp.sum(xraw*xraw))
    q = 2.*(x[0] > 0) - 1
    p = x[0]/q

    return jnp.concatenate(
        [-q*x[1:].reshape(1, -1),
         jnp.eye(x.shape[0]-1)-1/(1+p)*x[1:][:, None]@x[1:][None, :]])


def esqrtm(x):
    """ sqrtm by eigenvalue
    """
    ei, ev = jla.eigh(x)
    return ev@(jnp.sqrt(ei)[:, None]*ev.T)


def make_complement_basis(x):
    """ make complement basis of     
    """
    n, p = x.shape
    P = esqrtm(x[:p, :].T@x[:p, :])
    Q = jla.solve(P, x[:p, :].T).T
    return jnp.concatenate([- Q@x[p:, :].T,
                            jnp.eye(n-p) - x[p:, :]@jla.solve(P + jnp.eye(p), x[p:, :].T)],
                           axis=0)


def generate_symmetric_tensor(key, k, m):
    """Generating symmetric tensor size k,m
    """
    mat = jnp.full(tuple(m*[k]), jnp.nan)
    current_idx = jnp.zeros(m, dtype=int)
    active_i = m - 1
    tval, key = grand(key, (1,))
    mat = mat.at[tuple(current_idx)].set(tval)
    while True:
        if current_idx[active_i] < k - 1:
            current_idx[active_i] += 1
            if jnp.isnan(mat[tuple(current_idx)]):
                i_s = tuple(sorted(current_idx))
                if jnp.isnan(mat[i_s]):
                    tval, key = grand(key, (1,))
                    mat = mat.at[i_s].set(tval)
                    # print('Doing %s' % str(i_s))
                mat[tuple(current_idx)] = mat[i_s]
                # print('Doing %s' % str(current_idx))
        elif active_i == 0:
            break
        else:
            next_pos = jnp.where(current_idx[:active_i] < k-1)[0]
            if next_pos.shape[0] == 0:
                break
            current_idx[next_pos[-1]] += 1
            current_idx[next_pos[-1]+1:] = 0

            active_i = m - 1
            if jnp.isnan(mat[tuple(current_idx)]):
                i_s = tuple(sorted(current_idx))
                if jnp.isnan(mat[i_s]):
                    tval, key = grand(key, (1,))
                    mat = mat.at[i_s].set(tval)
                    # print('Doing %s' % str(i_s))
                mat = mat.at[tuple(current_idx)].set(mat[i_s])
                # print('Doing %s' % str(current_idx))
    return mat, key


def _fill_symmetric(p_raw, k):
    """Fill a k by k matrix with p_raw symmetrically
    """
    p = jnp.zeros((k, k))
    start = 0
    for i in range(k-1):
        p = p.at[i, i+1:].set(p_raw[start:start+k-i-1])
        p = p.at[i+1:, i].set(p_raw[start:start+k-i-1])
        start += k-i-1
    return jnp.fill_diagonal(p, p_raw[-k:], inplace=False)


def tv_mode_product(tensor, x, modes):
    """ Evaluating tensor subsituting x for the last modes times indices
    """
    v = tensor
    for _ in range(modes):
        v = jnp.tensordot(v, x, axes=1)
    return v


def _gen_so_inertia_matrix(key, n):
    """ generate an n times n symmetric matrix
    with diagonal 1
    """
    i_mat, key = grand(key, (n , n))
    i_mat = sym(jnp.abs(i_mat))
    i_mat = i_mat.at[jnp.diag_indices(n)].set(1.)
    return i_mat, key


def _old_rand_positive_definite(key, n):
    """ generate a positive definite matrix of size n
    """
    # n2 = (n*(n-1)) // 2
    mat, key = grand(key, (n, n))
    mat = mat@mat.T
    return sym(mat), key


def rand_positive_definite(key, n, bounds=None):
    """ generate a positive definite matrix of size n
    """
    # n2 = (n*(n-1)) // 2
    mat, key = grand(key, (n, n))
    if not bounds:
        return sym(mat@mat.T), key
    mat, _ = jla.qr(mat)
    key, sk = random.split(key)
    ei = random.uniform(sk, (n,), minval =bounds[0], maxval=bounds[1])
    mat = mat@(ei[:, None]*mat.T)
    return sym(mat), key


def _so_metric_opt(lu_mat, a):
    """ bilinear form given by lu_mat operates on a.
    lu_mat is a (n(n-1)/2)*(n(n-1)/2) matrix, operates on vectorization
    of the upper and lower triangular matrices so the overall operation is
    self-adjoint.
    """
    p = a.shape[0]
    rows, cols = jnp.triu_indices(p, 1)

    ret = jnp.empty((p, p))
    ret = ret.at[rows, cols].set(lu_mat@a.take(rows*p+cols))
    ret = ret.at[cols, rows].set(lu_mat@a.T.take(rows*p+cols))
    ret = ret.at[jnp.diag_indices(p)].set(a[jnp.diag_indices(p)])

    return ret


def _inv_so_metric_opt(lu_mat, a):
    """ invert of lu_mat
    """
    p = a.shape[0]
    rows, cols = jnp.triu_indices(p, 1)

    ret = jnp.empty((p, p))
    ret = ret.at[rows, cols].set(jla.solve(lu_mat, a.take(rows*p+cols)))
    # rows, cols = jnp.tril_indices(p, -1)
    ret = ret.at[cols, rows].set(jla.solve(lu_mat, a.T.take(rows*p+cols)))
    ret = ret.at[jnp.diag_indices(p)].set(a[jnp.diag_indices(p)])

    return ret


def unvec_skew(v):
    """ unravel a n(n-1)//2 vector to anti hermitian matrix
    """
    sqrt2 = jnp.sqrt(2.)
    rows = .5 * (1 + jnp.sqrt(1 + 8 * v.shape[0]))
    rows = jnp.round(rows).astype(int)
    result = jnp.zeros((rows, rows))
    result = result.at[jnp.triu_indices(rows, 1)].set(v)

    return (result.T - result)/sqrt2


def unvec_anti_hermitian(v):
    """ unravel a n(n-1)//2 vector to anti hermitian matrix
    """
    sqrt2 = jnp.sqrt(2)
    rows = .5 * (1 + jnp.sqrt(1 + 8 * len(v)))
    rows = int(jnp.round(rows))
    result = jnp.zeros((rows, rows))
    result = result.at[jnp.triu_indices(rows, 1)].set(v)
    return (result.T.conjugate() - result)/sqrt2


def unvech(v):
    """ Unvvectorize a symmetric matrix to a real vector
    Undoing the vech operation.
    sqrt2*upper triangular part concatenate with diagonal
    This is compatible with the trace(a@b) metric

    Parameters
    ----------
    v  : A vector
    Returns
    ----------
    the symmetric matrix undoing the vech operation
    """
    sqrt2 = jnp.sqrt(2)
    # quadratic formula, correct fp error
    rows = .5 * (-1 + jnp.sqrt(1 + 8 * v.shape[0]))
    rows = jnp.round(rows).astype(int)

    result = jnp.zeros((rows, rows))
    result = result.at[jnp.triu_indices(rows)].set(v/jnp.sqrt(2))
    result = result.at[jnp.diag_indices(rows)].set(
        result[jnp.diag_indices(rows)]/sqrt2)
    # result = (result + result.T)/sqrt2
    # divide diagonal elements by 2

    return result + result.T



def lie(a, b):
    """ Lie Bracket
    """
    return a@b - b@a


@jit
def jpolar(x):
    """ jax polar decomposition
    """
    return jla.solve(esqrtm(x.T@x), x.T).T
