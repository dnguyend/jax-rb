import numpy as np
from scipy.integrate import quad
from mpmath import jtheta


def k1(phi, t):
    """heat kernel on the circle
    """
    return 1/(2*np.pi)*float(jtheta(3, 0.5*phi, np.exp(-t), 0))


def shk1(x, y, t, d_coeff=.5):
    """heat kernel on the sphere s1.
    relationship with thk_1 below is thk1 are in radian (thus x, y there are scalars
    while x, y here are points on the cicle.
    x = (cos(x_torus), sin(x_torus))
    thus, np.sum(x*y) = cos(x_torus - y_torus)
    """
    return k1(np.arccos(np.sum(x*y)), t*d_coeff)


def thk1(x, y, t, d_coeff=0.5):
    """ fundamental solution to the heat equation on the torus, the interval (0, 2pi)
    d/dt f = d_coeff d/dx**2 f
    """
    return 1/(2*np.pi)*jtheta(3, 0.5*(y-x), np.exp(-d_coeff*t), 0)


def thk1a(x, y, t, d_coeff=0.5):
    """ alternative form
    """
    return 1/np.sqrt(4*d_coeff*np.pi*t)*np.exp(-0.25*(y-x)**2/t/d_coeff) \
        *jtheta(3, 0.5j*np.pi*(y-x)/t/d_coeff, np.exp(-np.pi**2/t/d_coeff), 0)


def thk1a1(x, y, t, d_coeff=0.5, N=10):
    """ summation form
    """
    s1 = 1
    ymx = y - x
    for ii in range(1, N):
        s1 += 2*np.exp(-d_coeff*ii**2*t)*np.cos(ii*ymx)
    return 1/(2*np.pi)*s1


def thk1a2(x, y, t, d_coeff=.5, N=10):
    """ summation form
    """
    s0 = 0
    pi2 = 2*np.pi
    ymx = y - x
    for ii in range(-N, N):
        s0 += np.exp(-(pi2*ii+ymx)**2/(4*d_coeff*t))
    return 1/np.sqrt(4*np.pi*d_coeff*t)*s0


def dy_thka2(x, y, t, d_coeff=.5, N=10):
    """ derivative in y
    """
    s0 = 0
    pi2 = 2*np.pi
    ymx = y - x
    for ii in range(-N, N):
        s0 += -2*(pi2*ii+ymx)/(4*d_coeff*t)*np.exp(-(pi2*ii+ymx)**2/(4*d_coeff*t))
    return 1/np.sqrt(4*np.pi*d_coeff*t)*s0


def k2(phi, t, N=10):
    """ heat kernel on the sphere S2
    using the Legendre polynomial - we implement the recursive
    formula for efficientcy
    """    
    # phi = np.abs(phi)
    # phi = phi - np.floor(phi/(np.pi))*np.pi
    # phi = np.arccos(np.cos(phi))
    
    cphi = np.cos(phi)
    pmv1 = cphi
    pmv2 = 1
    s = 1 + 3*np.exp(-2*t)*cphi
    for ix in range(2, N):
        pv = 1/ix*((2*ix-1)*pmv1*cphi - (ix-1)*pmv2)
        pmv2 = pmv1
        pmv1 = pv
        s += np.exp(-ix*(ix+1)*t)*(2*ix+1)*pv

    return  1/(4*np.pi)*s


def _k2_int(phi, t):
    return 1/2*quad(lambda u: k3(np.arccos(np.sin(u)*np.cos(phi/2)), t/4, True), -np.pi/2., np.pi/2)[0]


def _k2_int2(phi, t, N):
    phi = np.arccos(np.cos(phi))
    qret = 0.
    def one_term(w, ii):
        return np.exp(t/4)/np.sqrt(4*np.pi**3*t)* \
            2*(w+2*np.pi*ii)/t*np.exp(-(w + 2*np.pi*ii)**2/t)/(np.cos(phi) - np.cos(2*w))**.5
    for ii in range(-N, N):
        qret += quad(lambda w: one_term(w, ii), phi/2., np.pi-phi/2)[0]

    return 2**(-0.5)*qret


def k3(phi, t, use_inverse=False):
    """base kernel for d=3
    """
    if np.abs(phi) > 1e-6:
        if use_inverse:
            return -0.5*np.exp(t)/(2*np.pi)**2*float(
                (
                    -np.sqrt(np.pi/(t**3))*phi*np.exp(-0.25*(phi)**2/t) \
                    *jtheta(3, 0.5j*np.pi*(phi)/t, np.exp(-np.pi**2/t), 0).real \
                    - (np.pi/t)**1.5*np.exp(-0.25*(phi)**2/t) \
                    *float(jtheta(3, 0.5j*np.pi*(phi)/t, np.exp(-np.pi**2/t), 1).imag)
                )
                /np.sin(phi))
        return -0.5*np.exp(t)/(2*np.pi)**2*float(jtheta(3, 0.5*phi, np.exp(-t), 1)/np.sin(phi))
    return -0.25*np.exp(t)/(2*np.pi)**2*float(jtheta(3, 0.5*phi, np.exp(-t), 2)/np.cos(phi))


def old_k3(phi, t):
    """confirmed satisfies
    """
    if np.abs(phi) > 1e-6:
        return -0.5*np.exp(t)/(2*np.pi)**2*float(jtheta(3, 0.5*phi, np.exp(-t), 1)/np.sin(phi))
    return -0.25*np.exp(t)/(2*np.pi)**2*float(jtheta(3, 0.5*phi, np.exp(-t), 2)/np.cos(phi))


def shk3(x, y, t, d_coeff=0.5):
    return k3(np.arccos(np.sum(x*y)), t*d_coeff)


