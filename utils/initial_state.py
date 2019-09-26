import numpy as np
import mpnum as mp
import tmps
from tmps.utils import state_reduction_as_ndarray, convert, broadcast_number_ground_state, get_thermal_state
import time
from scipy.misc import factorial
import math


def get_spin_initial_state(theta, mpa_type='mps'):
    """
       Returns the initial state for the spin impurity:
       psi_0 = cos(theta) |1>  + sin(theta) |0>
       in the desired tensor network form (mps, mpo, pmps)
    """
    ground = np.array([0.0, np.sin(theta)])
    excited = np.array([np.cos(theta), 0.0])
    return convert.to_mparray(ground + excited, mpa_type)


def get_spin_boson_0T_chain_initial_state(theta, bath_local_dim, nof_coefficients):
    """
       Returns the full initial state (vacuum state) for 0T chain with nof_coefficients sites and a local dimension of
       bath_local_dim.
    """
    sys_psi_0 = get_spin_initial_state(theta)
    bath_psi_0 = broadcast_number_ground_state(bath_local_dim, nof_coefficients)
    return mp.chain([sys_psi_0, bath_psi_0])


def get_spin_boson_0T_star_initial_state(theta, system_index, bath_local_dim, nof_coefficients):
    """
       Returns the full initial state (vacuum state) for 0T star with nof_coefficients sites and a local dimension of
       bath_local_dim. The impurity is located at system_index.
    """
    sys_psi_0 = get_spin_initial_state(theta)
    # Initial states of the bath sites left and right of the system:
    left_bath_psi_0, right_bath_psi_0 = tmps.utils.broadcast_number_ground_state(bath_local_dim, system_index), \
                                        tmps.utils.broadcast_number_ground_state(bath_local_dim,
                                                                                 nof_coefficients - system_index)
    return mp.chain([left_bath_psi_0, sys_psi_0, right_bath_psi_0]
                    if left_bath_psi_0 is not None else [sys_psi_0, right_bath_psi_0])


def _compute_finiteT_chain_residual(psi_0, mpa_type, dims):
    """
        Returns residual of the finite-temperature initial state of the bath. List of populations in
        the highest energy state of each mode
    """
    res = []
    for index, dim in enumerate(dims):
        res.append(np.real(state_reduction_as_ndarray(psi_0, mpa_type, startsite=index)[dim - 1, dim - 1]))
    return res


def get_spin_boson_finiteT_chain_initial_state(theta, beta, h_site, h_bond, bath_local_dim, nof_coefficients,
                                               mpa_type='pmps',
                                               nof_steps=None, state_compression_kwargs=None,
                                               op_compression_kwargs=None, second_order_trotter=False,
                                               psi_0_compression_kwargs=None, residual=True,
                                               force_pmps_evolution=True, verbose=True):
    """
        Computes the initial state for the finite temperature spin_boson model in chain geometry.
        The bath state is computed via imaginary time evolution.
    :param theta: Spin parameter for  psi_0 = cos(theta) |1>  + sin(theta) |0>
    :param beta: Inverse temperature of the bath
    :param h_site: Bath local Hamiltonian list
    :param h_bond: Bath nearest neighbor coupling Hamiltonian list
    :param bath_local_dim: Local dimension of the bath
    :param nof_coefficients: Number of bath sites
    :param mpa_type: MPS type of the chain (mps, mpo, pmps)
    :param nof_steps: Number of steps for the imaginary time evolution
    :param state_compression_kwargs: Keyword args for the imaginary time evolution compression
    :param op_compression_kwargs: Keyword args for the imaginary time evolution operator pre-compression
    :param second_order_trotter: Set True for second order trotter based imaginary time evolution
    :param psi_0_compression_kwargs: Keyword args for the imaginary time evolution initial state compression
    :param residual: Set True to compute List of populations in the highest energy state of each bath mode.
    :param force_pmps_evolution: Set True to always use pmps for the imaginary time evolution
    :param verbose: Set true to make imaginary time evolution verbose
    :return: Initial state of system and bath as mps, mpo or pmps, info dict
    """
    assert mpa_type == 'mpo' or mpa_type == 'pmps'
    if nof_steps is None:
        nof_steps = int(beta*100)
    t0_wall = time.clock()
    t0_proc = time.perf_counter()
    if isinstance(bath_local_dim, int):
        dims = [bath_local_dim] * nof_coefficients
    else:
        raise AssertionError('Unsupported data type for fixed_dim')
    psi_0, info = tmps.chain.thermal.from_hamiltonian(beta, mpa_type, h_site, h_bond,
                                                      nof_steps=nof_steps,
                                                      state_compression_kwargs=state_compression_kwargs,
                                                      op_compression_kwargs=op_compression_kwargs,
                                                      second_order_trotter=second_order_trotter,
                                                      psi_0_compression_kwargs=psi_0_compression_kwargs,
                                                      force_pmps_evolution=force_pmps_evolution,
                                                      verbose=verbose)
    tf_proc = time.perf_counter() - t0_proc
    tf_wall = time.clock() - t0_wall
    info['walltime'] = tf_wall
    info['cpu_time'] = tf_proc
    info['bath_dims'] = dims
    if residual:
        res = _compute_finiteT_chain_residual(psi_0, mpa_type, dims)
        max_res = np.max(res)
        info['res'] = res
        info['max_res'] = max_res
    else:
        info['res'] = None
        info['max_res'] = None
    print('Finite T ground state residual ', info['res'])
    print('Finite T ground state max. residual: ', info['max_res'])
    sys_psi_0 = get_spin_initial_state(theta, mpa_type=mpa_type)
    return mp.chain([sys_psi_0, psi_0]), info


def get_star_local_dims(beta, xi, fixed_dim=None, high_energy_pop=1e-20, sitewise=False):
    """
        Computes the local dimension for the finite temperature star bath for the spin_boson model.
    :param beta: Inverse temperature of the bath
    :param xi: Star geometry bath energies
    :param fixed_dim: Uses this fixed dimension for the star evolution
    :param high_energy_pop: Chooses local dimension, such that the population in the highest energy of each bath mode
                            stays below this threshold
    :param sitewise: If set False the local dimension is chosen uniformly for all sites to be the
                     highest local dimension from the high_energy_pop calculation.
    :returns: List of bath dimensions
    """
    if fixed_dim is None:
        dims = []
        for xi_i in xi:
            a = 1 / (np.exp(beta * xi_i) - 1)
            dims.append(math.ceil(1 / (beta * xi_i) * np.log(1 + 1 / (high_energy_pop * a))))
        if sitewise:
            return dims
        else:
            return [np.max(dims)]*len(xi)
    else:
        if isinstance(fixed_dim, (list, tuple)):
            assert len(fixed_dim) == len(xi)
            return fixed_dim
        elif isinstance(fixed_dim, int):
            return [fixed_dim]*len(xi)
        else:
            raise AssertionError('Unsupported data type for fixed_dim')


def _compute_finite_T_star_residual(beta, xi, dims):
    """
        Returns residual of the finite-temperature initial state of the bath. List of populations in
        the highest energy state of each mode
    """
    res = []
    for xi_i, dim in zip(xi, dims):
        res.append((np.exp(beta*xi_i) - 1)/(np.exp(beta*xi_i * dim)))
    return res


def get_spin_boson_finiteT_star_initial_state(theta, beta, system_index, xi, mpa_type='pmps', fixed_dim=None,
                                              high_energy_pop=1e-20, sitewise=False, residual=True):
    """
            Computes the initial state for the finite temperature spin_boson model in star geometry.
        The bath state is computed via imaginary time evolution.
    :param theta: Spin parameter for  psi_0 = cos(theta) |1>  + sin(theta) |0>
    :param beta: Inverse temperature of the bath
    :param system_index: Impurity position in the auxiliary chain
    :param xi: Star geometry bath energies
    :param mpa_type: Type: mps, mpo or pmps of the initial state
    :param fixed_dim: Uses this fixed dimension for the star evolution
    :param high_energy_pop: Chooses local dimension, such that the population in the highest energy of each bath mode
                            stays below this threshold
    :param sitewise: If set False the local dimension is chosen uniformly for all sites to be the
                     highest local dimension from the high_energy_pop calculation.
    :param residual: Computes list of populations in the highest energy state of each mode
    :return: Initial state of system and bath as mps, mpo or pmps, info dict
    """
    assert mpa_type == 'mpo' or mpa_type == 'pmps'
    t0_wall = time.clock()
    t0_proc = time.perf_counter()
    dims = get_star_local_dims(beta, xi, fixed_dim=fixed_dim, high_energy_pop=high_energy_pop, sitewise=sitewise)
    ops = [xi[i] * np.arange(dim) for i, dim in enumerate(dims)]
    if system_index > 0:
        left_state = get_thermal_state(beta, mpa_type, ops[:system_index], to_cform=None)
        right_state = get_thermal_state(beta, mpa_type, ops[system_index:], to_cform=None)
    else:
        left_state = None
        right_state = get_thermal_state(beta, mpa_type, ops, to_cform=None)
    tf_proc = time.perf_counter() - t0_proc
    tf_wall = time.clock() - t0_wall
    info = dict()
    info['walltime'] = tf_wall
    info['cpu_time'] = tf_proc
    info['bath_dims'] = dims
    if residual:
        info['res'] = _compute_finite_T_star_residual(beta, xi, dims)
        info['max_res'] = np.max(info['res'])
    else:
        info['res'] = None
        info['max_res'] = None
    sys_psi_0 = get_spin_initial_state(theta, mpa_type=mpa_type)
    return mp.chain([left_state, sys_psi_0, right_state]) if left_state is not None else \
        mp.chain([sys_psi_0, right_state]), info


def get_boson_boson_0T_chain_initial_state(alpha, nof_coefficients, cutoff_dim):
    """
        Initial state for the Boson-Boson model in chain geometry (see Sec. 4.4.3 of the thesis)
    :param alpha: accuracy alpha for the impurity coherent state
    :param nof_coefficients: Number of bath sites
    :param cutoff_dim: Local dimension of the system and impurity
    :return: Initial state in MPS form
    """
    pop = lambda x: np.exp(-np.abs(alpha) ** 2 / 2) * alpha ** x / np.sqrt(factorial(x))
    sys_psi_0 = convert.to_mparray(pop(np.arange(cutoff_dim)), 'mps')
    bath_psi_0 = broadcast_number_ground_state(cutoff_dim, nof_coefficients)
    return mp.chain([sys_psi_0, bath_psi_0])


def get_boson_boson_0T_star_initial_state(alpha, system_index, nof_coefficients, cutoff_dim):
    """
        Initial state for the Boson-Boson model in star geometry (see Sec. 4.4.3 of the thesis)
    :param alpha: accuracy alpha for the impurity coherent state
    :param system_index: Index of the impurity in the auxiliary chain
    :param nof_coefficients: Number of bath sites
    :param cutoff_dim: Local dimension of the system and impurity
    :return: Initial state in MPS form
    """
    pop = lambda x: np.exp(-np.abs(alpha) ** 2 / 2) * alpha ** x / np.sqrt(factorial(x))
    sys_psi_0 = convert.to_mparray(pop(np.arange(cutoff_dim)), 'mps')
    # Initial states of the bath sites left and right of the system:
    left_bath_psi_0, right_bath_psi_0 = tmps.utils.broadcast_number_ground_state(cutoff_dim, system_index), \
                                        tmps.utils.broadcast_number_ground_state(cutoff_dim,
                                                                                 nof_coefficients - system_index)
    return mp.chain([left_bath_psi_0, sys_psi_0, right_bath_psi_0]
                    if left_bath_psi_0 is not None else [sys_psi_0, right_bath_psi_0])
