"""Module implementing simulation methods for embedded manifolds
"""
from functools import partial


import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnums=(0,))
def geodesic_move(mnf, x, unit_move, scale):
    """ simulate using a second order retraction.
    The move is :math:`x_{new} = \\mathfrak{r}(x, \\Pi(x)\\sigma(x)(\\text{unit_move}(\\text{scale})^{\\frac{1}{2}}))`
    """
    return mnf.retract(x, mnf.proj(x, mnf.sigma(x, unit_move.reshape(mnf.shape)*jnp.sqrt(scale))))


@partial(jit, static_argnums=(0,))
def geodesic_move_with_drift(mnf, x, unit_move, scale, additional_drift):
    """ This method is used to simulate a Riemanian Brownian motion with drift. The additional_drift
    is on top of the Brownian motion.
    Simulate using a second order retraction.
    The move is :math:`x_{new} = \\mathfrak{r}(x, \\Pi(x)\\sigma(x)(\\text{unit_move}(\\text{scale})^{\\frac{1}{2}}+\\text{scale}\\times\\text{additional_drift}))`
    """
    return mnf.retract(x, mnf.proj(x, mnf.sigma(x, unit_move.reshape(mnf.shape)*jnp.sqrt(scale))
                                   + scale*additional_drift))



@partial(jit, static_argnums=(0,))
def geodesic_move_normalized(mnf, x, unit_move, scale):
    """ similar to geodesic_move, but the move is normalized to have fixed length :math:`scale^{\\frac{1}{2}}`
    """
    # stochastic dx
    v = mnf.proj(x, mnf.sigma(x, unit_move.reshape(mnf.shape)))
    v = v/jnp.sqrt(mnf.inner(x, v, v))*jnp.sqrt(scale)
    return mnf.retract(x, v)


@partial(jit, static_argnums=(0,))
def geodesic_move_exact(mnf, x, unit_move, scale):
    """ similar to geodesic_move, but use exact geodesic
    """
    return mnf.exp(x, mnf.proj(x, mnf.sigma(x, unit_move.reshape(mnf.shape)*jnp.sqrt(scale))))


@partial(jit, static_argnums=(0,))
def geodesic_move_exact_normalized(mnf, x, unit_move, scale):
    """ similar to geodesic_move_exact, but use normalize the unit_move
    """
    # stochastic dx
    v = mnf.proj(x, mnf.sigma(x, unit_move.reshape(mnf.shape)))
    v = v/jnp.sqrt(mnf.inner(x, v, v))*jnp.sqrt(scale)
    return mnf.exp(x, v)


@partial(jit, static_argnums=(0,))
def rbrownian_ito_move(mnf, x, unit_move, scale):
    """ 
    Use Euler Maruyama and projection method to solve the Ito equation.
    """
    return mnf.approx_nearest(
        x + mnf.proj(x, mnf.sigma(x, unit_move.reshape(mnf.shape)*jnp.sqrt(scale)))
        + mnf.ito_drift(x)*scale)


@partial(jit, static_argnums=(0,))
def ito_move_with_drift(mnf, x, unit_move, scale, additional_drift):
    """ This method is used to simulate a Riemanian Brownian motion with drift. The additional_drift
    is on top of the Brownian motion.
    Use Euler Maruyama and projection method to solve the Ito equation.
    """
    return mnf.approx_nearest(
        x + mnf.proj(x, mnf.sigma(x, unit_move.reshape(mnf.shape)*jnp.sqrt(scale)))
        + (additional_drift+mnf.ito_drift(x))*scale)


@partial(jit, static_argnums=(0,))
def rbrownian_stratonovich_move(mnf, x, unit_move, scale):
    """ Use Euler Heun and projection method to solve the Stratonovich equation.
    """
    # stochastic dx
    dxs = mnf.sigma(x, unit_move.reshape(mnf.shape)*jnp.sqrt(scale))
    xbk = x + mnf.proj(x, dxs)
    return mnf.approx_nearest(x + mnf.proj(0.5*(x + xbk), dxs)
                              + mnf.proj(x, mnf.ito_drift(x)*scale))


@partial(jit, static_argnums=(0,))
def stratonovich_move_with_drift(mnf, x, unit_move, scale, additional_drift):
    """
    This method is used to simulate a Riemanian Brownian motion with drift. The additional_drift
    is on top of the Brownian motion.
    Use Euler Heun and projection method to solve the Stratonovich equation.
    """
    # stochastic dx
    dxs = mnf.sigma(x, unit_move.reshape(mnf.shape)*jnp.sqrt(scale))
    xbk = x + mnf.proj(x, dxs)
    return mnf.approx_nearest(x + mnf.proj(0.5*(x + xbk), dxs)
                              + mnf.proj(x, mnf.ito_drift(x)+additional_drift)*scale)
