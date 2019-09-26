import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def get_two_gaussians(N=0.5, mu=(1, 3), b=(100, 100), r=(1, 1)):
    """
        Returns the two Gaussians spectral density defined in Sec. 4.2 of the thesis
    """
    mu1 = mu[0]
    mu2 = mu[1]
    b1 = b[0]
    b2 = b[1]
    r1 = r[0]
    r2 = r[1]
    alpha = N / (r1 * np.sqrt(np.pi / b1) + r2 * np.sqrt(np.pi / b2))
    a1 = r1 * alpha
    a2 = r2 * alpha
    return lambda x: a1 * np.exp(-b1 * (x - mu1)**2) + a2 * np.exp(-b2 * (x - mu2)**2)


def get_ohmic(N=0.5, m=0.8, s=1):
    """
        Returns the Ohmic spectral density defined in Sec. 4.2 of the thesis
    """
    omega_c = m / s
    alpha = N / (omega_c**2 * gamma(s + 1))
    return lambda x: alpha * omega_c * (x / omega_c) ** s * np.exp(-x / omega_c)


def get_owp(N=0.5, m=1, s=1, I=0.3, gam=0.15, x0=3.5, domain=(0, 25)):
    """
        Returns the Ohmic with peak spectral density defined in Sec. 4.2 of the thesis
    """
    N, m, s, omega_c, I, gam, x0, domain = \
        N, m, s, m / s, I, gam, x0, domain
    J_not_normalized = lambda x: (I * gam ** 2 / ((x - x0) ** 2 + gam ** 2)) * np.exp(-np.abs(x - x0) / gam) + \
                                 omega_c * (x / omega_c) ** s * np.exp(-x / omega_c)
    alpha = N / quad(J_not_normalized, a=domain[0], b=domain[1], epsabs=1e-13, epsrel=1e-13, limit=100)[0]
    J = lambda x: alpha * (I * gam ** 2 / ((x - x0) ** 2 + gam ** 2)) * np.exp(-np.abs(x - x0) / gam) \
                  + omega_c * (x / omega_c) ** s * np.exp(-x / omega_c)
    return J


def get_semi_elliptical(N=0.5, domain=(0, 4)):
    """
        Returns the semi-elliptical spectral density defined in Sec. 4.2 of the thesis
    """
    mu = (domain[0] + domain[1])/2
    a = (domain[1] - domain[0])/2
    alpha = N / (np.pi*a / 2)
    return lambda x: alpha * np.sqrt(1 - ((x-mu)/a)**2)