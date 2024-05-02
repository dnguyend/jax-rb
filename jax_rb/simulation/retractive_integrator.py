"""Module implementing the retractive Euler-Maruyama integrator.
"""
from functools import partial


import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnums=(0,2,5,6))
def retractive_move(rtr, x, t, unit_move, scale, sigma, mu):
    """ Simulating the equation :math:`dX_t = \\mu(X_t, t) dt + \\sigma(X_t, t) dW_t` using the retraction rtr.
    We do not assume a Riemanian metric on the manifold, :math:`\\sigma\\sigma^T` could be degenerated on :math:`T\\mathcal{M}`.

    W is a Wiener process driving the equation, defined on :math:`\\mathbb{R}^k`. W is given by unit_move.

    :math:`\\sigma(X_t, t)` maps :math:`\\mathbb{R}^k` to :math:`\\mathcal{E}`, but the image belongs 
    to :math:`T_{X_t}\\mathcal{M}`.

    The retraction rtr is assume to have the method :math:`\\text{drift_adj}` for an adjustment.
    
    The move is :math:`x_{new} = \\mathfrak{r}(x, \\Pi(x)\\sigma(x)(\\text{unit_move}(\\text{scale})^{\\frac{1}{2}}) + \\text{scale} (\\mu + \\text{drift_adj}))`.

    :param rtr: the retraction,
    :param x: a point on the manifold,
    :param t: time
    :param unit_move: a random normal draw
    :param scale: scaling
    :param sigma: a function implementing the map :math:`\\sigma`
    :param mu: a function implementing the Ito drift :math:`\\mu`
    """
    return rtr.retract(x,
                       sigma(x, t, unit_move)*jnp.sqrt(scale)
                       + scale*(mu(x, t) + rtr.drift_adjust(sigma, x, t, unit_move.shape[0])))


@partial(jit, static_argnums=(0,2,5,6))
def retractive_move_normalized(rtr, x, t, unit_move, scale, sigma, mu):
    """ Similar to retractive_move, but the stochastic part is normalized to have fixed length :math:`scale^{\\frac{1}{2}}`
    """
    # v = mnf.proj(x, mnf.sigma(x, unit_move.reshape(mnf.shape)))
    # v = v/jnp.sqrt(mnf.inner(x, v, v))*jnp.sqrt(scale)
    # return mnf.retract(x, v)
    v = sigma(x, t, unit_move)
    mnf = rtr.mnf
    return rtr.retract(x,
                       sigma(x, t, v/jnp.sqrt(mnf.inner(x, v, v))*jnp.sqrt(scale*mnf.dim))
                       + scale*(mu(x, t) + rtr.drift_adjust(sigma, x, t, unit_move.shape[0])))
