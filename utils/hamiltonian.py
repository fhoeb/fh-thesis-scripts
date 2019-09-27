from scipy.misc import factorial
from itertools import count
import numpy as np
from tmps.utils import pauli, fock


def get_boson_boson_dim(alpha, cutoff_coh):
    """
        Find the cutoff for the local dimension (identical everywhere) from the chosen accuracy alpha for the impurity
        coherent state.
    """
    #
    pop = lambda x: np.exp(-np.abs(alpha) ** 2 / 2) * alpha ** x / np.sqrt(factorial(x))
    cutoff_dim = 2
    for n in count(cutoff_dim, 1):
        if np.abs(pop(n))**2 < cutoff_coh:
            cutoff_dim = n
            break
    return cutoff_dim


def get_spin_boson_chain_hamiltonian(omega_0, c0, omega, t, bath_local_dim, finite_T=False):
    """
        Returns local and coupling parts of the Spin-Boson model chain Hamiltonian
        used in Sec. 4.4.1 and 4.4.2 of the thesis.
    :param omega_0: Spin energy
    :param c0: Spin-Bath coupling
    :param omega: Bath energies
    :param t: Bath-bath couplings
    :param bath_local_dim: Local dimension of the bath
    :param finite_T: If set True builds the Hamiltonian for Sec. 4.4.2. If False builds the Hamiltonian for Sec. 4.4.1
    :returns: List of local Hamiltonians, List of coupling Hamiltonians
    """
    if not finite_T:
        # Local Hamiltonian of the System:
        spin_loc = omega_0 / 2 * pauli.X

        # Coupling between System and bath:
        spin_coupl = pauli.Z
    else:
        # Local Hamiltonian of the System:
        spin_loc = omega_0 / 2 * pauli.Z

        # Coupling between System and bath:
        spin_coupl = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    # Local Hamiltonian of the bath
    fock_n = fock.n(bath_local_dim)
    bath_loc = [energy * fock_n for energy in omega]

    # Bath coupling
    bath_coupling_op = np.kron(fock.a(bath_local_dim), fock.a_dag(bath_local_dim)) + \
                       np.kron(fock.a_dag(bath_local_dim), fock.a(bath_local_dim))
    bath_bath_coupl = [coupling * bath_coupling_op for coupling in t]

    # Spin-Bath coupling
    spin_bath_coupl = c0 * (np.kron(spin_coupl, fock.a_dag(bath_local_dim)) +
                            np.kron(spin_coupl.conj().T, fock.a(bath_local_dim)))
    return [spin_loc] + bath_loc, [spin_bath_coupl] + bath_bath_coupl


def get_spin_boson_star_hamiltonian(omega_0, system_index, gamma, xi, bath_local_dim, finite_T=False):
    """
        Returns local and coupling parts of the Spin-Boson model star Hamiltonian
        used in Sec. 4.4.1 and 4.4.2 of the thesis.
    :param omega_0: Spin energy
    :param system_index: Index of the system in the auxiliary chain
    :param gamma: System-Bath couplings
    :param xi: Bath energies
    :param bath_local_dim: Local dimension of the bath
    :param finite_T: If set True uses the Hamiltonian for Sec. 4.4.2. If False builds the Hamiltonian for Sec. 4.4.1
    :returns: List of local Hamiltonians, List of coupling Hamiltonians
    """
    if not finite_T:
        # Local Hamiltonian of the System:
        spin_loc = omega_0 / 2 * pauli.X

        # Coupling between System and bath:
        spin_coupl = pauli.Z
    else:
        # Local Hamiltonian of the System:
        spin_loc = omega_0 / 2 * pauli.Z

        # Coupling between System and bath:
        spin_coupl = np.array([[0, 0], [1, 0]], dtype=np.complex128)

    # Local Hamiltonian of the bath
    fock_n = fock.n(bath_local_dim)
    bath_loc = [energy * fock_n for energy in xi]

    # Coupling operators for the bath to the left of the system
    left_bath_coupling_op = np.kron(fock.a(bath_local_dim), spin_coupl.conj().T) + \
                            np.kron(fock.a_dag(bath_local_dim), spin_coupl)
    left_bath_coupl = [coupling * left_bath_coupling_op for coupling in gamma[:system_index]]
    # Coupling operators for the bath to the right of the system
    right_bath_coupling_op = np.kron(spin_coupl.conj().T, fock.a(bath_local_dim)) + \
                             np.kron(spin_coupl, fock.a_dag(bath_local_dim))
    right_bath_coupl = [coupling * right_bath_coupling_op for coupling in gamma[system_index:]]
    return bath_loc[:system_index] + [spin_loc] + bath_loc[system_index:], left_bath_coupl + right_bath_coupl


def get_boson_boson_chain_hamiltonian(omega_0, c0, omega, t, cutoff_dim):
    """
        Returns local and coupling parts of the Spin-Boson model chain Hamiltonian
        used in Sec. 4.4.3 of the thesis.
    :param omega_0: Spin energy
    :param c0: Spin-Bath coupling
    :param omega: Bath energies
    :param t: Bath-bath couplings
    :param cutoff_dim: Local dimension of the impurity and bath
    :returns: List of local Hamiltonians, List of coupling Hamiltonians
    """
    # Local Hamiltonian of the System:
    sys_loc = omega_0 * fock.n(cutoff_dim)

    # Coupling between System and bath:
    sys_coupl = fock.a(cutoff_dim)

    # Local Hamiltonian of the bath
    fock_n = fock.n(cutoff_dim)
    bath_loc = [energy * fock_n for energy in omega]

    # Bath coupling
    bath_coupling_op = np.kron(fock.a(cutoff_dim), fock.a_dag(cutoff_dim)) + \
                       np.kron(fock.a_dag(cutoff_dim), fock.a(cutoff_dim))
    bath_bath_coupl = [coupling * bath_coupling_op for coupling in t]

    # Spin-Bath coupling
    spin_bath_coupl = c0 * (np.kron(sys_coupl, fock.a_dag(cutoff_dim)) +
                            np.kron(sys_coupl.conj().T, fock.a(cutoff_dim)))
    return [sys_loc] + bath_loc, [spin_bath_coupl] + bath_bath_coupl


def get_boson_boson_star_hamiltonian(omega_0, system_index, gamma, xi, cutoff_dim):
    """
        Returns local and coupling parts of the Spin-Boson model star Hamiltonian
        used in Sec. 4.4.3 of the thesis.
    :param omega_0: Spin energy
    :param system_index: Index of the system in the auxiliary chain
    :param gamma: System-Bath couplings
    :param xi: Bath energies
    :param cutoff_dim: Local dimension of the impurity and bath
    :returns: List of local Hamiltonians, List of coupling Hamiltonians
    """
    # Local Hamiltonian of the System:
    sys_loc = omega_0 * fock.n(cutoff_dim)

    # Coupling between System and bath:
    sys_coupl = fock.a(cutoff_dim)

    # Local Hamiltonian of the bath
    fock_n = fock.n(cutoff_dim)
    bath_loc = [energy * fock_n for energy in xi]

    # Coupling operators for the bath to the left of the system
    left_bath_coupling_op = np.kron(fock.a(cutoff_dim), sys_coupl.conj().T) + \
                            np.kron(fock.a_dag(cutoff_dim), sys_coupl)
    left_bath_coupl = [coupling * left_bath_coupling_op for coupling in gamma[:system_index]]
    # Coupling operators for the bath to the right of the system
    right_bath_coupling_op = np.kron(sys_coupl.conj().T, fock.a(cutoff_dim)) + \
                             np.kron(sys_coupl, fock.a_dag(cutoff_dim))
    right_bath_coupl = [coupling * right_bath_coupling_op for coupling in gamma[system_index:]]
    return bath_loc[:system_index] + [sys_loc] + bath_loc[system_index:], left_bath_coupl + right_bath_coupl
