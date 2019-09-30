import tmps
import numpy as np
from utils.discretization import get_discretized_coefficients
from utils.hamiltonian import get_boson_boson_chain_hamiltonian, get_boson_boson_star_hamiltonian, get_boson_boson_dim
from utils.initial_state import get_boson_boson_0T_chain_initial_state, get_boson_boson_0T_star_initial_state
from utils.logger import Logger
from utils.plotter import multiplot
from utils.residual import compute_boson_boson_residual
from os.path import join
import subprocess

# System path which must exist which will contain the subfolder with the data
root = '.'
# Name of the subfolder which will contain the data
data = 'BB_Ohmic_0.5_0.8_1'

# General parameters
# Type of spectrall density. Possible are 'ohmic', 'ohmic_with_peak', 'semi_elliptical', 'two_gaussians'
spectral_density = 'ohmic'
# Parameters for the spectral densities (see thesis for names)
spectral_density_params = {'N': 0.5, 'm': 0.8, 's': 1}
# Domain of the spectral density
domain = [0, 20]

# Spin energy
omega_0 = 1

# Location of the system in the star
star_system_index = 0

# Number of bath sites
nof_bath_sites = 30


# Discretization parameters:

# Log the discretization (False means nohing will be logged)
disc_log = True

# Discretization type,  'bsdo' or 'linear'
disc_type = 'bsdo'

# Orthpol accuracy parameter
bsdo_ncap = 10000

# Number of linear discretization intervals
lin_ncap = None

# Star to chain mapping type. Use sp_hes for stability and few coefficients. Use lan_bath for many coefficients
# (i.e. high lin_ncap) at possible cost of stability
mapping_type = 'sp_hes'


# Time evolution parameters

# Time evolution (if set False, no time evolution will be performed)
time_evolve = True

# Log frequency (must be >= 0, 0 means no logging)
te_log_frequency = 1

# Plot the results at the end and save the plot
te_plot = True

# Open the plot directly in a pdf viewer
show_plot = True

# Verbose output
verbose = True

# Timestep
tau = 0.01

# Number of steps
nof_steps = 100

# State compression arguments:
state_compression = {'method': 'svd', 'relerr': 1e-7}

# Trotter-decomposition operator precompression
op_compression = {'method': 'svd', 'relerr': 1e-10, 'stable': True}

# Trotter decomposition kind (if set False, fourth order is used)
second_order_trotter = False

# Compression of the initial state before time evolution
psi_0_compression = None


# Initial state parameters:

# Parameter for the initial state Psi_0 = cos(theta) |1>  + sin(theta) |0>
spin_theta = 0

# Local dimension of the system and each bath site is calculated from alpha and cutoff_coh

# Coherent state parameter
alpha = 1.8
# Threshold for the population in the highest energy fock basis state of the coherent state
cutoff_coh = 1e-8

local_dim = get_boson_boson_dim(alpha, cutoff_coh)

print('System and bath local dimension: ', local_dim)


# Initialize logger

logger = Logger(data, root=root)


# Obtain discretized coefficients for the chain (c0 [system-bath coupling], omega [bath energies],
# t [bath-bath couplings])
# and for the star (gamma [system-bath couplings, xi [bath energies])
print('Calculating discretized coefficients')
(c0, omega, t), (gamma, xi) = \
    get_discretized_coefficients(nof_bath_sites, spectral_density, spectral_density_params,
                                 domain, disc_type, bsdo_ncap=bsdo_ncap,
                                 lin_ncap=lin_ncap, mapping_type=mapping_type)

if disc_log:
    logger.log_coeff(chain_potentials=omega, chain_couplings=np.append(np.array([c0]), t),
                     star_potentials=xi, star_couplings=gamma)


if time_evolve:
    # Obtain the parts which make up the chain and star Hamiltonians
    print('Calculating the Hamiltonian')
    chain_h_site, chain_h_bond = get_boson_boson_chain_hamiltonian(omega_0, c0, omega, t, local_dim)
    star_h_site, star_h_bond = get_boson_boson_star_hamiltonian(omega_0, star_system_index, gamma, xi, local_dim)

    # Obtain the initial state of system and bath
    print('Calculating the initial states of star and chain')
    chain_psi_0 = get_boson_boson_0T_chain_initial_state(alpha, nof_bath_sites, local_dim)
    star_psi_0 = get_boson_boson_0T_star_initial_state(alpha, star_system_index, nof_bath_sites, local_dim)

    mpa_type = 'mps'

    # Generating propagators
    print('Preparing time evolution for star and chain')
    chain_propagator = tmps.chain.from_hamiltonian(chain_psi_0, mpa_type, chain_h_site, chain_h_bond,
                                                   tau=tau, state_compression_kwargs=state_compression,
                                                   op_compression_kwargs=op_compression,
                                                   second_order_trotter=second_order_trotter,
                                                   psi_0_compression_kwargs=psi_0_compression)

    star_propagator = tmps.star.from_hamiltonian(star_psi_0, mpa_type, star_system_index, star_h_site, star_h_bond,
                                                 tau=tau, state_compression_kwargs=state_compression,
                                                 op_compression_kwargs=op_compression,
                                                 second_order_trotter=second_order_trotter,
                                                 psi_0_compression_kwargs=psi_0_compression)

    # Discrete simulation times
    times = tau * np.arange(0, nof_steps+1)

    # Prepare results arrays
    chain_ranks = np.empty((nof_steps + 1, nof_bath_sites))
    star_ranks = np.empty((nof_steps + 1, nof_bath_sites))
    chain_size = np.empty(nof_steps + 1)
    star_size = np.empty(nof_steps + 1)
    rel_diff = np.empty(nof_steps + 1)

    # Fill results arrays with values from initial state
    chain_ranks[0][:] = chain_propagator.psi_t.ranks
    star_ranks[0][:] = star_propagator.psi_t.ranks
    chain_size[0] = chain_propagator.psi_t.size
    star_size[0] = star_propagator.psi_t.size
    rel_diff[0] = compute_boson_boson_residual(chain_propagator.psi_t, 0, star_propagator.psi_t, star_system_index,
                                               mpa_type)

    if te_log_frequency > 0:
        logger.log_timeevo(chain_ranks=chain_ranks[:1], star_ranks=star_ranks[:1],
                           chain_size=chain_size[:1], star_size=star_size[:1],
                           rel_diff=rel_diff[:1])

    # Main propagation loop
    print('Starting time evolution')
    for step in range(1, nof_steps+1):
        print('Computing step: ', step)
        chain_propagator.evolve()
        star_propagator.evolve()

        chain_ranks[step][:] = chain_propagator.psi_t.ranks
        star_ranks[step][:] = star_propagator.psi_t.ranks
        chain_size[step] = chain_propagator.psi_t.size
        star_size[step] = star_propagator.psi_t.size
        rel_diff[step] = compute_boson_boson_residual(chain_propagator.psi_t, 0, star_propagator.psi_t,
                                                      star_system_index, mpa_type)

        if verbose:
            print('Chain size: ' + str(chain_size[step]) + '. Star size: ' + str(star_size[step]))
            print('Current chain ranks: ', chain_ranks[step][:])
            print('Current star ranks: ', star_ranks[step][:])
            print('Relative difference: ', rel_diff[step])

        if step % te_log_frequency == 0:
            print('Logging')
            logger.log_timeevo(chain_ranks=chain_ranks[:step], star_ranks=star_ranks[:step],
                               chain_size=chain_size[:step], star_size=star_size[:step],
                               rel_diff=rel_diff[:step])

    multiplot(logger.root, 'overview', nof_bath_sites, times, chain_ranks, star_ranks, chain_size, star_size,
              rel_diff)

    if show_plot:
        subprocess.Popen([join(logger.root, 'overview.pdf')], shell=True)
