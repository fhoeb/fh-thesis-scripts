from utils.spectral_densities import get_ohmic, get_owp, get_two_gaussians, get_semi_elliptical
import mapping as map


def get_discretized_coefficients(nof_coefficients, sd_type, sd_kwargs, domain, disc_type, bsdo_ncap=30000,
                                 lin_ncap=None, mapping_type='sp_hes'):
    """
        Calculates the coefficients for the discretized star and chain Hamiltonians
    :param nof_coefficients: Number of coefficients to calculate
    :param sd_type: Type of spectral density: 'ohmic', 'ohmic_with_peak', 'two_gaussians' or 'semi_elliptical'
                    (see Sec. 4.2 of the thesis)
    :param sd_kwargs: Keyword arguments for the parameters of the specified spectral density. See the
                      spectral_densities module and Sec. 4.2 of the thesis.
    :param domain: Support of the spectral density
    :param disc_type: Discretization type: 'linear' or 'bsdo' (see Sec. 2.2 of the thesis)
    :param bsdo_ncap: Accuracy parameter for py-orthpol for the bsdo coefficients
    :param lin_ncap: Accuracy parameter for the linear discretization, see py-mapping library. None means no
                     high-resolution star is used.
    :param mapping_type: Type of tridiagonalization if required. See py-mapping library.
    :return: Tuple of chain and star coefficients: ((c0, omega, t), (gamma, xi)) where xi denotes the star energies.
    """
    if sd_type == 'ohmic':
        J = get_ohmic(**sd_kwargs)
    elif sd_type == 'ohmic_with_peak':
        J = get_owp(**sd_kwargs, domain=domain)
    elif sd_type == 'two_gaussians':
        J = get_two_gaussians(**sd_kwargs)
    elif sd_type == 'semi_elliptical':
        J = get_semi_elliptical(**sd_kwargs, domain=domain)
    else:
        raise AssertionError('Unsupported spectral density type')

    if disc_type == 'linear':
        disc_type = 'sp_quad'
        ncap = lin_ncap
    elif disc_type == 'bsdo':
        ncap = bsdo_ncap
    else:
        raise AssertionError('Unsupported discretization type')

    coeff_options = {'ignore_zeros': True, 'low_memory': True, 'stable': False, 'quad_order': 40,
                     'epsrel': 1e-13, 'epsabs': 1e-13, 'limit': 100, 'force_sp': False, 'mp_dps': 20,
                     'permute': None, 'ncap': ncap}
    c0, omega, t, info = map.chain.get(J, domain, nof_coefficients, disc_type=disc_type,
                                       interval_type='lin', mapping_type=mapping_type, **coeff_options)
    gamma, xi = map.star.get(J, domain, nof_coefficients=nof_coefficients, disc_type=disc_type,
                             interval_type='lin', **coeff_options)
    return (c0, omega, t), (gamma, xi)


