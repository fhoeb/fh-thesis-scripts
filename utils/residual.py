import numpy as np
from tmps.utils import state_reduction_as_array


def compute_spin_boson_residual(phi, phi_system_index, psi, psi_system_index, mpa_type):
    """
        Calculates the residual, see Eq. 4.25 in the thesis
    :param phi: Star or Chain state
    :param phi_system_index: Index of the system for phi
    :param psi: Star or Chain state with which to compare phi
    :param phi_system_index: Index of the system for psi
    :param mpa_type: mps, mpo or pmps
    :return: residual (float)
    """
    phi_sys = state_reduction_as_array(phi, mpa_type, phi_system_index, nof_sites=1)
    psi_sys = state_reduction_as_array(psi, mpa_type, psi_system_index, nof_sites=1)
    return np.linalg.norm(phi_sys - psi_sys) / np.linalg.norm(phi_sys)


def compute_boson_boson_residual(phi, phi_system_index, psi, psi_system_index, mpa_type):
    """
        Calculates the residual, see Eq. 4.32 in the thesis
    :param phi: Star or Chain state
    :param phi_system_index: Index of the system for phi
    :param psi: Star or Chain state with which to compare phi
    :param phi_system_index: Index of the system for psi
    :param mpa_type: mps, mpo or pmps
    :return: residual (float)
    """
    phi_sys_pop = np.abs(np.diag(state_reduction_as_array(phi, mpa_type, phi_system_index, nof_sites=1))).copy()
    phi_sys_exp = np.sum(np.arange(len(phi_sys_pop)) * phi_sys_pop)

    psi_sys_pop = np.abs(np.diag(state_reduction_as_array(psi, mpa_type, psi_system_index, nof_sites=1))).copy()
    psi_sys_exp = np.sum(np.arange(len(psi_sys_pop)) * psi_sys_pop)
    return np.abs(phi_sys_exp - psi_sys_exp) / np.abs(phi_sys_exp)
