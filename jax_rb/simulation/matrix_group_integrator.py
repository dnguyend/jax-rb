"""Module implementing simulation methods for left invariant matrix Lie group
"""
from functools import partial


import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnums=(0,))
def geodesic_move(mnf, x, unit_move, scale):
    """ :math:`\\text{unit_move}` is reshaped to the shape conforming with sigma., usually the shape of the ambient space.
    The move is :math:`x_{new} = \\mathfrak{r}(x, \\sigma(x)(\\text{unit_move}(\\text{scale})^{\\frac{1}{2}}))`
    """
    return x@mnf.retract(jnp.eye(mnf.shape[0]),
                         mnf.sigma_id(
                             jnp.sqrt(scale)*unit_move.reshape(mnf.shape)))


@partial(jit, static_argnums=(0,))
def geodesic_move_with_drift(mnf, x, unit_move, scale, id_additional_drift):
    """ This method is used to simulate a Riemanian Brownian motion with drift.
    :math:`\\text{unit_move}` is reshaped to the shape conforming with sigma., usually the shape of the ambient space. :math:`\\text{id_additional_drift}` is an element of the Lie algebra.
    The move is :math:`x_{new} = \\mathfrak{r}(x, \\sigma(x)((\\text{scale})^{\\frac{1}{2}}\\times \\text{unit_move})+\\text{scale}\\times x (\\text{id_additional_drift}))`
    """
    return x@mnf.retract(jnp.eye(mnf.shape[0]),
                         mnf.sigma_id(
                             jnp.sqrt(scale)*unit_move.reshape(mnf.shape))
                         + id_additional_drift*scale)


@partial(jit, static_argnums=(0,))
def geodesic_move_normalized(mnf, x, unit_move, scale):
    """ Similar to geodesic_move, but unit move is rescaled to have fixed length 1
    in the metric of the group.
    """
    v = mnf.sigma_id(unit_move.reshape(mnf.shape))
    v = v / jnp.sqrt(mnf.inner(jnp.eye(mnf.shape[0]), v, v)) * jnp.sqrt(scale)
    return x@mnf.retract(jnp.eye(mnf.shape[0]), v)


@partial(jit, static_argnums=(0,))
def geodesic_move_dim_g(mnf, x, unit_move, scale):
    """:math:`\\text{unit_move}` is of dimension :math:`\\dim \\mathrm{G}`.
    The move is :math:`x_{new} = \\mathfrak{r}(x, \\sigma_{la}(x)(\\text{unit_move}(\\text{scale})^{\\frac{1}{2}}))`
    """
    return x@mnf.retract(jnp.eye(mnf.shape[0]),
                         mnf.sigma_la(jnp.sqrt(scale)*unit_move))


@partial(jit, static_argnums=(0,))
def geodesic_move_dim_g_normalized(mnf, x, unit_move, scale):
    """ Similar to geodesic_move_dim_g, but unit move is rescaled to have fixed length 1
    in the metric of the group.
    """
    nu = unit_move/jnp.sqrt(jnp.sum(unit_move**2))
    return x@mnf.retract(jnp.eye(mnf.shape[0]),
                         mnf.sigma_la(jnp.sqrt(scale)*nu))


@partial(jit, static_argnums=(0,))
def rbrownian_ito_move(mnf, x, unit_move, scale):
    """ Use stochastic projection method to solve the Ito equation.
    Use Euler Maruyama here.
    """
    n = mnf.shape[0]
    return mnf.approx_nearest(
        x@jnp.eye(n) + x@mnf.sigma_id(unit_move.reshape(mnf.shape)*jnp.sqrt(scale))
        + x@mnf.id_drift*scale)


@partial(jit, static_argnums=(0,))
def ito_move_with_drift(mnf, x, unit_move, scale, id_additional_drift):
    """ This method is used to simulate a Riemanian Brownian motion with drift given in Ito form. Use stochastic projection method to solve the Ito equation.
    The drift is given as an element of the Lie algebra.
    Use Euler Maruyama here.
    """
    n = mnf.shape[0]
    return mnf.approx_nearest(
        x@jnp.eye(n) + x@mnf.sigma_id(unit_move.reshape(mnf.shape)*jnp.sqrt(scale))
        + x@mnf.id_drift*scale + x@id_additional_drift*scale)


@partial(jit, static_argnums=(0,))
def rbrownian_stratonovich_move(mnf, x, unit_move, scale):
    """ Using projection method to solve the Stratonovich equation.
    In many cases :math:`v_0` is zero (unimodular group).
    Use Euler Heun.
    """
    n = mnf.shape[0]
    # stochastic dx
    dxs = mnf.sigma_id(unit_move.reshape(mnf.shape)*jnp.sqrt(scale))

    move = jnp.eye(n) + 0.5*(2*jnp.eye(n)+dxs)@dxs + mnf.v0*scale
    return x@mnf.approx_nearest(move)


@partial(jit, static_argnums=(0,))
def stratonovich_move_with_drift(mnf, x, unit_move, scale, id_additional_drift):
    """ This method is used to simulate a Riemanian Brownian motion with drift given in Stratonovich form.
    Using projection method to solve the Stratonovich equation.
    The additional drift is on top of the RB term, given as an element of the Lie algebra
    In many cases :math:`v_0` is zero (unimodular group).
    Use Euler Heun.
    """
    n = mnf.shape[0]
    # stochastic dx
    dxs = mnf.sigma_id(unit_move.reshape(mnf.shape)*jnp.sqrt(scale))

    move = jnp.eye(n) + 0.5*(2*jnp.eye(n)+dxs)@dxs + mnf.v0*scale + scale*id_additional_drift
    return x@mnf.approx_nearest(move)


@partial(jit, static_argnums=(0,))
def ito_move_dim_g(mnf, x, unit_move, scale):
    """Similar to rbrownian_ito_move, but driven with a Wiener
    process of dimension :math:`\\dim \\mathrm{G}`.
    """
    return x@mnf.approx_nearest(
        jnp.eye(mnf.shape[0]) + mnf.sigma_la(unit_move*jnp.sqrt(scale))
        + mnf.id_drift*scale)

@partial(jit, static_argnums=(0,))
def stratonovich_move_dim_g(mnf, x, unit_move, scale):
    """Similar to rbrownian_stratonovich_move, but driven with a Wiener
    process of dimension :math:`\\dim \\mathrm{G}`. 
    """
    n = mnf.shape[0]
    # stochastic dx
    dxs = mnf.sigma_la(unit_move*jnp.sqrt(scale))

    move = jnp.eye(n) + 0.5*(2*jnp.eye(n)+dxs)@dxs + mnf.v0*scale
    return x@mnf.approx_nearest(move)
