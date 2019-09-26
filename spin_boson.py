import tmps
import numpy as np
from utils.discretization import get_discretized_coefficients
from utils.hamiltonian import get_spin_boson_chain_hamiltonian, get_spin_boson_star_hamiltonian
from utils.initial_state import get_spin_boson_0T_chain_initial_state, get_spin_boson_0T_star_initial_state, \
    get_spin_boson_finiteT_chain_initial_state, get_spin_boson_finiteT_star_initial_state
from utils.logger import Logger
from utils.plotter import multiplot
from utils.residual import compute_spin_boson_residual
from os.path import join
import subprocess

# System path which must exist which will contain the subfolder with the data
root = '.'
# Name of the subfolder which will contain the data (must exist)
data = 'SB_Ohmic_2_0.8_1'

# General parameters
# Type of spectrall density. Possible are 'ohmic', 'ohmic_with_peak', 'semi_elliptical', 'two_gaussians'
spectral_density = 'ohmic'
# Parameters for the spectral densities (see thesis for names)
spectral_density_params = {'N': 2, 'm': 0.8, 's': 1}
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

# Discretization type, 'bsdo' or 'linear'
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

# Parameter for the initial state psi_0 = cos(theta) |1>  + sin(theta) |0>
spin_theta = 0

# Inverse temperature of the bath; set to np.inf for vacuum state
beta = np.inf

# Local dimension of each bath site for the star and the chain
chain_local_dim = 14
star_local_dim = 14

# Propagation mpa-type for finite temperature ('pmps' or 'mpo')
ft_mpa_type = 'pmps'

# Chain imaginary time evolution parameters:
# Number of steps
ite_nof_steps = 100
# State compression arguments:
ite_state_compression = {'method': 'svd', 'relerr': 1e-4}
# Trotter-decomposition operator precompression
ite_op_compression = {'method': 'svd', 'relerr': 1e-8, 'stable': True}
# Trotter decomposition kind (if set False, fourth order is used)
ite_second_order_trotter = False
# Compression of the initial state before imag. time evolution
ite_psi_0_compression = None
# Obtain residual (popualation in the highest energy state of each bath mode) from the imaginary time evolution
chain_residual = True
# Use pmps for the imaginary time evolution regardless of chosen ft_mpa_type
ite_force_pmps_evolution = True
# Set verbose output for the imaginary time evolution
ite_verbose = True

# Star thermal state parameters:
# Overrides star_local_dim. Chooses local dim such that population in the highest energy mode for each bath site
# stays below this threshold
high_energy_pop = 1e-12
# Chooses local dimension for each bath mode individually (when using high_energy_pop). If set False the highest value
# for star_local dim for any of the modes is used for all modes.
sitewise = False
# Obtain residual (popualation in the highest energy state of each bath mode)
star_residual = True


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
    print('Building the Hamiltonian')
    chain_h_site, chain_h_bond = get_spin_boson_chain_hamiltonian(omega_0, c0, omega, t, chain_local_dim,
                                                                  finite_T=np.isfinite(beta))
    star_h_site, star_h_bond = get_spin_boson_star_hamiltonian(omega_0, star_system_index, gamma, xi, star_local_dim,
                                                               finite_T=np.isfinite(beta))

    # Obtain the initial state of system and bath
    print('Calculating the initial states of star and chain')
    if np.isinf(beta):
        chain_psi_0 = get_spin_boson_0T_chain_initial_state(spin_theta, chain_local_dim, nof_bath_sites)
        star_psi_0 = get_spin_boson_0T_star_initial_state(spin_theta, star_system_index, star_local_dim, nof_bath_sites)
    else:
        print('Performing imaginary time evolution for the initial state of the chain bath')
        chain_psi_0, chain_ft_info = \
            get_spin_boson_finiteT_chain_initial_state(spin_theta, beta, chain_h_site[1:], chain_h_bond[1:],
                                                       chain_local_dim, nof_bath_sites, mpa_type=ft_mpa_type,
                                                       nof_steps=ite_nof_steps,
                                                       state_compression_kwargs=ite_state_compression,
                                                       op_compression_kwargs=ite_op_compression,
                                                       second_order_trotter=ite_second_order_trotter,
                                                       psi_0_compression_kwargs=ite_psi_0_compression,
                                                       residual=chain_residual,
                                                       force_pmps_evolution=ite_force_pmps_evolution,
                                                       verbose=ite_verbose)
        logger.log_metadata('chain_finiteT_info.txt', chain_ft_info)
        star_psi_0, star_ft_info = \
            get_spin_boson_finiteT_star_initial_state(spin_theta, beta, star_system_index, xi, mpa_type=ft_mpa_type,
                                                      fixed_dim=star_local_dim, high_energy_pop=high_energy_pop,
                                                      sitewise=sitewise, residual=star_residual)
        logger.log_metadata('star_finiteT_info.txt', star_ft_info)

    # Set time evolution mpa_type
    if np.isinf(beta):
        mpa_type = 'mps'
    else:
        mpa_type = ft_mpa_type

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
    rel_diff[0] = compute_spin_boson_residual(chain_propagator.psi_t, 0, star_propagator.psi_t, star_system_index,
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
        rel_diff[step] = compute_spin_boson_residual(chain_propagator.psi_t, 0, star_propagator.psi_t,
                                                     star_system_index, mpa_type)

        if verbose:
            print('Chain size: ' + str(int(chain_size[step])) + '. Star size: ' + str(int(star_size[step])))
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
