from blockymetamaterials.energy import constrain_energy
from blockymetamaterials.geometry import DOFsInfo, Geometry, compute_inertia
from blockymetamaterials.kinematics import build_constrained_kinematics
from blockymetamaterials.utils import ControlParams, LigamentParams, MechanicalParams, load_data, SolutionType

import numpy as np
import matplotlib.colors as colors
import jax.numpy as jnp
from typing import Any, Callable, NamedTuple, Union, Optional
from sympy import symbols, lambdify, diff, I

import scipy
from scipy.integrate import solve_ivp
from jax import hessian, vmap

import subprocess
def say(text):
    subprocess.call(['say', text])


class LatticeParams(NamedTuple):
    n1_blocks : Any
    n2_blocks : Any
    spacing : Any
    bond_length : Any
    initial_angle : Any

class SimParams(NamedTuple):
    lattice_params : LatticeParams
    mechanical_params : MechanicalParams

def experimental_parameters():
    exp_config = {
        'n1_blocks': 24,
        'n2_blocks': 16,
        'spacing': 10.0,
        'hinge_length': 0.5,
        'density': 8.82964e-9,  # Mg/mm^2
        
        # Mechanical parameters
        'k_stretch': 20.58824,
        'k_shear': 0.82941,
        'k_rot': 1.68,
        
        # Damping coefficients
        'n_stretch': 0.76536e-3,  # Mg/s
        'n_shear': 0.76536e-3,    # Mg/s
        'n_rot': 2.67277e-3,      # Mg.mm^2/s
        'damptype': 'hinge',
    }

    # Number of blocks N
    exp_config['N'] = exp_config['n1_blocks'] * exp_config['n2_blocks']

    return exp_config

""" STRUCTURE OF PICKLED DICTIONARY OBJECT:
datadict = {
    "data" : data,
    "params" : SimParams(
        lattice_params = LatticeParams(
            n1_blocks=geometry.n1_blocks,
            n2_blocks=geometry.n2_blocks,
            spacing=geometry.spacing,
            bond_length=geometry.bond_length,
            initial_angle=initial_angle
        ),
        mechanical_params = MechanicalParams(
            bond_params=StretchingTorsionalSpringParams(
                k_stretch=k_stretch,
                k_rot=k_rot,
            ),
            density=density,
        ),
    )
}
"""

def unpack(path):
    pickle_dict = load_data(path)
    return pickle_dict["data"], pickle_dict["params"].lattice_params, pickle_dict["params"].mechanical_params


# CREATE A LIGHTER SHADE OF COLOR
def adjust_lightness(color, amount=0.5):
    """
    Adjusts the lightness of a color.
    
    Parameters:
    color: matplotlib color (name, hex, RGB tuple, etc.)
    amount: float where:
        > 0 = lighten (0.5 = 50% lighter)
        < 0 = darken (-0.5 = 50% darker)
        0 = no change
    
    Returns:
    RGB tuple of the adjusted color
    """
    try:
        c = colors.to_rgb(color)
    except:
        c = color
    
    if amount > 0:
        # Lighten: mix with white
        return tuple(c[i] + (1 - c[i]) * amount for i in range(3))
    else:
        # Darken: mix with black
        return tuple(c[i] * (1 + amount) for i in range(3))

# ==============================================================================
# LINEAR MODEL ANALYSIS FUNCTIONS
# ==============================================================================

# CLAMPING CONSTRAINTS
def constraint_info(n1_blocks, n2_blocks, clamping=True):
    """  Returns the indices of free and clamped degrees of freedom (DOFs) in a periodic lattice.
    
    Parameters:
    - n1_blocks: Number of blocks in the first direction.
    - n2_blocks: Number of blocks in the second direction.
    - clamping: If True, returns indices for clamped DOFs.
    
    Returns:
    - free_DOF_ids: Indices of free DOFs.
    - clamped_DOF_ids: Indices of clamped DOFs.
    - all_DOF_ids: All DOF indices.
    """

    N = n1_blocks * n2_blocks

    clampedblocks = [] if clamping is None else [0, 1, n1_blocks-2, n1_blocks-1, n1_blocks, 2*n1_blocks-1, N-n1_blocks-1, N-2, N-1, N-2*n1_blocks, N-n1_blocks, N-n1_blocks+1]
    clamped_block_DOF_pairs = np.array([np.array([block_id, DOF_id]) for DOF_id in range(3) for block_id in clampedblocks])

    # Retrieve free DOFs from constraints info
    free_DOF_ids, clamped_DOF_ids, all_DOF_ids = DOFsInfo(N, clamped_block_DOF_pairs)

    return free_DOF_ids, clamped_DOF_ids, all_DOF_ids


def linear_mode_analysis(
        displacement: np.ndarray,
        geometry: Geometry,
        energy_fn: Callable,
        control_params: ControlParams,
        constrained_block_DOF_pairs: np.ndarray,
        return_force_coeffs: bool = False,
        only_eigs: bool = False,):
    """Computes eigenvalues and eigenmodes for a conservative lattice system K @ q = ω^2 M @ q.

    Args:
        displacement (jnp.ndarray): configuration around which linearization is performed.
        geometry (Geometry): Geometry of the structure.
        energy_fn (Callable): Potential energy functional.
        centroid_node_vectors (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes.
        inertia (Union[jnp.ndarray, float]): either a scalar or an array of shape (n_blocks, 3) collecting the inertia of the blocks.
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        tuple: eigenvalues and eigenmodes. The eigenmodes are returned as an array of shape (n_modes, n_blocks, 3)
    """

    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Retrieve free DOFs from constraints info
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Reduce displacement and inertia to the free DOFs
    _displacement = displacement.reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    if control_params.mechanical_params.inertia is None:
        _inertia = compute_inertia(
            vertices=control_params.geometrical_params.centroid_node_vectors,
            density=control_params.mechanical_params.density
        ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    else:
        _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

    stiffness_matrix = hessian(constrained_energy)(_displacement, 0, control_params)

    if return_force_coeffs==True:
        return stiffness_matrix

    # eigenvectors given by scipy are organized column-wise
    if only_eigs is True:
        eigenvalues = scipy.linalg.eigvalsh(
            stiffness_matrix,
            jnp.diag(_inertia),
        )
        return np.array(eigenvalues)

    eigenvalues, eigenvectors = scipy.linalg.eigh(
        stiffness_matrix,
        jnp.diag(_inertia),
    )  # jnp.linalg.eigh does not currently implement generalized eigenvalue problems
    # Normalize and transpose eigenvectors
    eigenvectors = vmap(lambda v: v / jnp.linalg.norm(v))(eigenvectors.T)

    # Reshape eigenvectors to global state. all_DOFs_modes are organized row-wise.
    all_DOFs_modes = jnp.zeros((len(free_DOF_ids), len(all_DOF_ids)))
    all_DOFs_modes = all_DOFs_modes.at[:, free_DOF_ids].set(
        eigenvectors
    )

    # NOTE: this returns eigenfrequency-squared and eigenmodes
    return np.array(eigenvalues), all_DOFs_modes.reshape((len(free_DOF_ids), geometry.n_blocks, 3))


def linear_softmode_analysis(
        displacement: np.ndarray,
        geometry: Geometry,
        energy_fn: Callable,
        control_params: ControlParams,
        constrained_block_DOF_pairs: np.ndarray,
        return_force_coeffs: bool = False,
        n_modes: int = 5,):
    """Computes smallest n eigenvalues and eigenmodes of K @ q = ω^2 M @ q.

    Args:
        displacement (jnp.ndarray): configuration around which linearization is performed.
        geometry (Geometry): Geometry of the structure.
        energy_fn (Callable): Potential energy functional.
        centroid_node_vectors (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes.
        inertia (Union[jnp.ndarray, float]): either a scalar or an array of shape (n_blocks, 3) collecting the inertia of the blocks.
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).
        n_modes (int, optional): Number of modes to be found, starting from the lowest eigenvalue (default is 5).

    Returns:
        tuple: eigenvalues and eigenmodes. The eigenmodes are returned as an array of shape (n_modes, n_blocks, 3)
    """

    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Retrieve free DOFs from constraints info
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Reduce displacement and inertia to the free DOFs
    _displacement = displacement.reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    if control_params.mechanical_params.inertia is None:
        mass_mat = compute_inertia(
            vertices=control_params.geometrical_params.centroid_node_vectors,
            density=control_params.mechanical_params.density
        ).reshape((geometry.n_blocks * 3,))
    else:
        mass_mat = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)

    _inertia = mass_mat[free_DOF_ids]
    print(_inertia[:6])

    stiffness_matrix = hessian(constrained_energy)(_displacement, 0, control_params)

    if return_force_coeffs==True:
        return stiffness_matrix

    stiffness_sparse = scipy.sparse.csr_matrix(stiffness_matrix)
    mass_sparse = scipy.sparse.csr_matrix(np.diag(_inertia))

    # eigenvectors given by scipy are organized column-wise    
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        stiffness_sparse,
        M = mass_sparse, k=n_modes, which='SM'
    )  # jnp.linalg.eigh does not currently implement generalized eigenvalue problems

    # Normalize and transpose eigenvectors
    eigenvectors = vmap(lambda v: v / jnp.linalg.norm(v))(eigenvectors.T)

    # Reshape eigenvectors to global state. all_DOFs_modes are organized row-wise.
    all_DOFs_modes = jnp.zeros((n_modes, len(all_DOF_ids)))
    all_DOFs_modes = all_DOFs_modes.at[:, free_DOF_ids].set(
        eigenvectors
    )

    # NOTE: return eigenfrequency squared and modes
    return np.array(eigenvalues), all_DOFs_modes.reshape((n_modes, geometry.n_blocks, 3))

def linear_dampedmode_analysis(
        displacement: np.ndarray,
        geometry: Geometry,
        energy_fn: Callable,
        control_params: ControlParams,
        constrained_block_DOF_pairs: np.ndarray,
        damp_params: ControlParams,
        damptype: str = 'hinge',
        return_force_coeffs: bool = False,
        only_eigs: bool = False,):
    """Computes eigenvalues and eigenmodes of disspative system K @ q + λ C @ q = - λ^2 M @ q.

    Args:
        displacement (jnp.ndarray): configuration around which linearization is performed.
        geometry (Geometry): Geometry of the structure.
        energy_fn (Callable): Potential energy functional.
        centroid_node_vectors (jnp.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes.
        inertia (Union[jnp.ndarray, float]): either a scalar or an array of shape (n_blocks, 3) collecting the inertia of the blocks.
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        tuple: eigenvalues and eigenmodes. The eigenmodes are returned as an array of shape (n_modes, n_blocks, 3)
    """

    # Handle constraints
    kinematics = build_constrained_kinematics(geometry=geometry, constrained_block_DOF_pairs=constrained_block_DOF_pairs)
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Retrieve free DOFs from constraints info
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Reduce displacement and inertia to the free DOFs
    _displacement = displacement.reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    if control_params.mechanical_params.inertia is None:
        _inertia = compute_inertia(
            vertices=control_params.geometrical_params.centroid_node_vectors,
            density=control_params.mechanical_params.density
        ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    else:
        _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

    stiffness_matrix = hessian(constrained_energy)(_displacement, 0, control_params)

    # Damping
    if damptype == 'hinge':
        # Compute the damping matrix using the hessian of the constrained energy
        damping_matrix = hessian(constrained_energy)(_displacement, 0, damp_params)
    elif damptype is None or damptype == 'none':
        damping_matrix = jnp.zeros_like(stiffness_matrix)

    if return_force_coeffs==True:
        return stiffness_matrix, damping_matrix


    # eigenvectors given by scipy are organized column-wise
    if only_eigs is True:
        eigenvalues = scipy.linalg.eigvalsh(
            stiffness_matrix,
            jnp.diag(_inertia),
        )
        return jnp.array(eigenvalues)
    
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        stiffness_matrix,
        jnp.diag(_inertia),
    )  # jnp.linalg.eigh does not currently implement generalized eigenvalue problems
    # Normalize and transpose eigenvectors
    eigenvectors = vmap(lambda v: v / jnp.linalg.norm(v))(eigenvectors.T)

    # Reshape eigenvectors to global state. all_DOFs_modes are organized row-wise.
    all_DOFs_modes = jnp.zeros((len(free_DOF_ids), len(all_DOF_ids)))
    all_DOFs_modes = all_DOFs_modes.at[:, free_DOF_ids].set(
        eigenvectors
    )

    # NOTE: returns complex eigenfrequencies I*(ω+I*γ) where ω is the oscillatory part and γ is the decay part
    return jnp.array(eigenvalues), all_DOFs_modes.reshape((len(free_DOF_ids), geometry.n_blocks, 3))



def linear_response_analysis(
        frequencies: np.ndarray,
        displacement: np.ndarray,
        geometry: Geometry,
        energy_fn: Callable,
        control_params: ControlParams,
        drive_amplitude: float,
        drive_angle: float,
        return_force_coeffs: bool = False,
        constrained_block_DOF_pairs: np.ndarray = np.array([]),
        driven_block_DOF_pairs: np.ndarray = np.array([]),
        damp_coeffs: np.ndarray = np.array([]),
        damptype: str = 'viscous'):
    """Computes solutions of driven dissipative system: 
        - K @ q - iw C @ q + P^T @ λ = - w^2 M @ q,     where drive contraints are P @ q = d, and λ is the vector of Lagrange multipliers. 
        Response of free DOFs is q exp[iwt].

    Args:
        frequencies (np.ndarray): frequencies to sweep through.
        displacement (np.ndarray): configuration around which linearization is performed.
        geometry (Geometry): Geometry of the structure.
        energy_fn (Callable): Potential energy functional.
        centroid_node_vectors (np.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes.
        inertia (Union[np.ndarray, float]): either a scalar or an array of shape (n_blocks, 3) collecting the inertia of the blocks.
        constrained_block_DOF_pairs (np.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).
        damptype (str, optional): Describes damping model (internal, external or non-existent). Defaults to 'viscous' (external).
        damp_coeffs (np.ndarray, optional): Coefficents for external damping model (when damptype='viscous').
        damp_params (ControlParams, optional): Parameters for internal damping model (when damptype='hinge').

    Returns:
        np.ndarray: Displacement response is returned as an array of shape (len(frequencies), n_blocks, 3)
    """

    print('computing constrained energy')
    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Retrieve free DOFs from constraints info
    free_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # Reduce displacement and inertia to the free DOFs
    _displacement = displacement.reshape((geometry.n_blocks * 3,))[free_DOF_ids]

    print('computing stiffness matrix')
    stiffness_matrix = hessian(constrained_energy)(_displacement, 0, control_params)

    # Damping
    print('computing damping matrix')
    if damptype == 'hinge':
        reference_bond_vectors = geometry.get_parametrization()[3]
        damp_params = ControlParams(
            geometrical_params=control_params.geometrical_params,
            mechanical_params=MechanicalParams(
                bond_params=LigamentParams(
                    k_stretch=damp_coeffs[0],
                    k_shear=damp_coeffs[1],
                    k_rot=damp_coeffs[2],
                    reference_vector=control_params.mechanical_params.bond_params.reference_vector
                ),
                density=control_params.mechanical_params.density,
            ),
            )
        _damping = hessian(constrained_energy)(_displacement, 0, damp_params)
    elif damptype == 'viscous': 
        _damping = jnp.diag(jnp.reshape(damp_coeffs*jnp.ones((geometry.n_blocks,3)),-1)[free_DOF_ids])
    else: _damping = jnp.diag(jnp.zeros(len(free_DOF_ids)))

    if return_force_coeffs == True:
        print('returning matrices')
        return stiffness_matrix, _damping
    
    # Mass matrix
    if control_params.mechanical_params.inertia is None:
        _inertia = compute_inertia(
            vertices=control_params.geometrical_params.centroid_node_vectors,
            density=control_params.mechanical_params.density
        ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    else:
        _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

    # Driving
    driven_DOF_ids = jnp.array([block_id * 3 + DOF_id for block_id, DOF_id in driven_block_DOF_pairs])
    print('Driven DOF ids : ', driven_DOF_ids)
    mat_P = jnp.zeros((len(driven_DOF_ids), geometry.n_blocks * 3))
    for i in range(len(driven_DOF_ids)):
        mat_P = mat_P.at[(i,driven_DOF_ids[i])].set(1)
    mat_P = jnp.delete(mat_P, constrained_DOF_ids, 1)
    mat_PT = jnp.transpose(mat_P)
    print(np.linalg.matrix_rank(mat_P))

    d = jnp.array([drive_amplitude*jnp.cos(drive_angle) if i%3 == 0 else drive_amplitude*jnp.sin(drive_angle) for i in driven_DOF_ids])
    print('Drive : ', d)
    b = jnp.concatenate((jnp.zeros(len(free_DOF_ids)), d))
    block4 = jnp.zeros((len(driven_DOF_ids),len(driven_DOF_ids)))

    print(f"No. of clamped dofs: {len(constrained_DOF_ids)}")
    print(f"No. of driven dofs: {len(driven_DOF_ids)}")
    print(f"b: {b}  ", f"b shape: {b.shape}")

    solset = jnp.empty((len(frequencies),len(free_DOF_ids)),dtype = np.complex128)
    j = 0
    
    for w in frequencies:
        block1 = (w**2) * jnp.diag(_inertia) - 1j*w*_damping - stiffness_matrix
        dynamic_matrix = jnp.block([[block1, - mat_PT], [mat_P, block4]])

        solution = scipy.linalg.solve(
            dynamic_matrix, b
        )
        
        solset = solset.at[j,:].set(solution[:len(free_DOF_ids)])
        j += 1
    
    print(f"solution set shape {solset.shape}")

    # Reshape eigenvectors to global state. all_DOFs_modes are organized row-wise.
    all_DOFs_modes = jnp.zeros((len(frequencies),len(all_DOF_ids)),dtype = np.complex128)
    all_DOFs_modes = all_DOFs_modes.at[:,free_DOF_ids].set(solset)

    return all_DOFs_modes.reshape((len(frequencies),geometry.n_blocks, 3))


def linear_timelocalised_response_analysis(
        drive_disp,
        drive_vel,
        drive_angle : float,
        drive_amplitude : float,
        displacement: np.ndarray,
        geometry: Geometry,
        energy_fn: Callable,
        control_params: ControlParams,
        timepoints: np.ndarray,
        constrained_block_DOF_pairs: np.ndarray = np.array([]),
        driven_block_DOF_pairs: np.ndarray = np.array([]),
        damptype: str = 'viscous',
        damp_coeffs: np.ndarray = np.array([]),
        damp_params: Optional[ControlParams] = None,
        give_K_N: bool = False,
        give_driveforce: bool = False):
    """Computes solutions of - Kf @ qf - Cf @ dqf/dt + f(t) = Mf @ d2qf/dt2,
        where f(t) = - Kfc @ d(t) - Cfc @ dd(t)/dt is the force applied by the driven DOFs on the free DOFs.

    Args:
        drive_disp (scalar function in time): imposed displacement by drive.
        drive_vel (scalar function in time): imposed velocity by drive.
        drive_angle (float) : angle from x-axis of drive-displacement direction.
        drive_amplitude (float) : amplitude of drive to be used for adjusting numerical tolerance in solver.
        displacement (np.ndarray): configuration around which linearization is performed.
        geometry (Geometry): Geometry of the structure.
        energy_fn (Callable): Potential energy functional.
        damptype (str, optional): Describes damping model ---'hinge' (internal) OR 'viscous' (external or non-existent). Defaults to 'viscous' (external).
        damp_coeffs (np.ndarray, optional): Coefficents for external damping model (when damptype='viscous').
        damp_params (ControlParams, optional): Parameters for internal damping model (when damptype='hinge').
        timepoints (np.ndarray): Return response values for this given time series. 
        give_K_N (bool, optional): To return stiffness and damping matrices.
        give_driveforce (bool, optional): To return time-dependent function that outputs force applied by driven DOFs on free DOFs.
        centroid_node_vectors (np.ndarray): array of shape (n_blocks, n_nodes_per_block, 2) representing the vectors connecting the centroid of the blocks to the nodes.
        inertia (Union[np.ndarray, float]): either a scalar or an array of shape (n_blocks, 3) collecting the inertia of the blocks.
        constrained_block_DOF_pairs (np.ndarray, optional): Only zero-valued DOF pairs. Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).
        driven_block_DOF_pairs (np.ndarray, optional): Only driven DOF pairs. Array of shape (n_driven, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        np.ndarray: The displacement and velocity response is returned as an array of shape (len(timepoints), 2, n_blocks, 3), 
    """

    # Handle constraints
    kinematics = build_constrained_kinematics(
        geometry=geometry,
        constrained_block_DOF_pairs=constrained_block_DOF_pairs
    )
    constrained_energy = constrain_energy(energy_fn=energy_fn, constrained_kinematics=kinematics)

    # Retrieve free DOFs from constraints info
    driven_DOF_ids = np.array([block_id * 3 + DOF_id for block_id, DOF_id in driven_block_DOF_pairs])
    unconstrained_DOF_ids, constrained_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    id_map = {v: i for i, v in enumerate(np.array(unconstrained_DOF_ids))}
    driven_ids_wrt_unconstrainedids = np.array([id_map[i] for i in driven_DOF_ids])

    # Reduce displacement and inertia to the free DOFs
    _displacement = displacement.reshape((geometry.n_blocks * 3,))[unconstrained_DOF_ids]

    free_DOF_ids, clamped_DOF_ids, all_DOF_ids = DOFsInfo(geometry.n_blocks, np.concatenate((constrained_block_DOF_pairs, driven_block_DOF_pairs)))
    
    if control_params.mechanical_params.inertia is None:
        _inertia = compute_inertia(
            vertices=control_params.geometrical_params.centroid_node_vectors,
            density=control_params.mechanical_params.density
        ).reshape((geometry.n_blocks * 3,))[free_DOF_ids]
    else:
        _inertia = control_params.mechanical_params.inertia.reshape(geometry.n_blocks * 3)[free_DOF_ids]

    stiffness_matrix = hessian(constrained_energy)(_displacement, 0, control_params)

    # Damping
    if damptype == 'hinge':
        _damping = hessian(constrained_energy)(_displacement, 0, damp_params)
        print('hinge damping!')
    elif damptype == 'viscous': 
        _damping = jnp.diag(jnp.reshape(damp_coeffs*jnp.ones((geometry.n_blocks,3)),-1)[unconstrained_DOF_ids])
        print('viscous damping!')
    else:
        _damping = jnp.diag(jnp.zeros(len(unconstrained_DOF_ids)))
        print('damping is zero!')

    if give_K_N == True:
        return stiffness_matrix, _damping


    free_ids_wrt_unconstrainedids = np.setdiff1d(np.arange(stiffness_matrix.shape[0]), driven_ids_wrt_unconstrainedids, assume_unique=True)
    print('free dofs : ', len(free_ids_wrt_unconstrainedids))

    # Define the forcing function
    def force(t):
        disp_vector = np.array([drive_disp(t)*np.cos(drive_angle) if i%3 == 0 else drive_disp(t)*np.sin(drive_angle) for i in driven_DOF_ids])
        vel_vector = np.array([drive_vel(t)*np.cos(drive_angle) if i%3 == 0 else drive_vel(t)*np.sin(drive_angle) for i in driven_DOF_ids])
        driventofree_stiffness_matrix = stiffness_matrix[np.ix_(free_ids_wrt_unconstrainedids, driven_ids_wrt_unconstrainedids)]
        driventofree_damping_matrix = _damping[np.ix_(free_ids_wrt_unconstrainedids, driven_ids_wrt_unconstrainedids)]
        f = - driventofree_stiffness_matrix @ disp_vector - driventofree_damping_matrix @ vel_vector
        return f.flatten()

    if give_driveforce==True:
        return force

    # Convert second-order ODE to first-order system
    def driven_system(t, q):
        x = q[:len(free_ids_wrt_unconstrainedids)]
        v = q[len(free_ids_wrt_unconstrainedids):]         # dx/dt
        dxdt = v
        dvdt = np.linalg.solve(np.diag(_inertia), force(t) - _damping[np.ix_(free_ids_wrt_unconstrainedids, free_ids_wrt_unconstrainedids)]  @ v - stiffness_matrix[np.ix_(free_ids_wrt_unconstrainedids, free_ids_wrt_unconstrainedids)] @ x)
        return np.concatenate([dxdt, dvdt])

    # Initial conditions: q(0) = [x(0), v(0)]
    x0 = np.zeros(len(free_ids_wrt_unconstrainedids))
    v0 = np.zeros(len(free_ids_wrt_unconstrainedids))
    q0 = np.concatenate([x0, v0])

    # Solve the system
    global sol
    sol = solve_ivp(driven_system, (timepoints[0],2*timepoints[-1]-timepoints[-2]), q0, t_eval=timepoints, method='RK45', rtol=1e-6, atol=1e-7 * drive_amplitude)
    print(f"solution set shape : ", sol.y.shape)

    drive_disp_field = np.array([[drive_disp(t)*np.cos(drive_angle) if i%3 == 0 else drive_disp(t)*np.sin(drive_angle) for i in driven_DOF_ids] for t in timepoints])
    drive_vel_field = np.array([[drive_vel(t)*np.cos(drive_angle) if i%3 == 0 else drive_vel(t)*np.sin(drive_angle) for i in driven_DOF_ids] for t in timepoints])

    fields = np.zeros((len(timepoints), 2, len(all_DOF_ids)))
    fields[:,0,driven_DOF_ids] = drive_disp_field
    fields[:,1,driven_DOF_ids] = drive_vel_field
    fields[:,:,free_DOF_ids] = sol.y.T.reshape((len(timepoints),2,-1))

    return fields.reshape((len(timepoints), 2, geometry.n_blocks, 3))


def response_funcn(coeffs, modes, lamdas, timepoints):
    """ Computes the response of the system to an initial state defined by eigenmode coefficients.
        Args:
            coeffs: Coefficients for the modes.
            modes: Modes of the system.
            lamdas: Eigenvalues corresponding to the modes.
            timepoints: Time points at which to evaluate the response.
        Returns:
            System's response as evolved for the given timepoints based on the eigenvalues (lamdas)
    """
    phase = lamdas[:, np.newaxis] * timepoints[np.newaxis, :]
    response_coeffs = coeffs[:, np.newaxis] * modes

    return np.einsum('it,ij->tj', np.exp(phase), response_coeffs, optimize='greedy')



# ==============================================================================
# LINEAR STRAIN & ELASTIC MODULI CALCULATION
# ==============================================================================

# Convert block displacements to unit cell displacements
def unitcell_data(data, geometry = None, mode_range = None, n1_blocks=0, n2_blocks=0):

    """ Takes in SolutionType, geometry and mode_range, and spits out ndarray of unit cell displacements for those modes
    """

    frames = range(len(data.fields)) if mode_range is None else mode_range
    if geometry is not None:
        n1_blocks = geometry.n1_blocks 
        n2_blocks = geometry.n2_blocks

    # organise data as a grid: [r,c] denotes block on column c of row r
    # block_centroids = data.block_centroids.reshape((n2_blocks, n1_blocks, 2))
    block_disps = data.fields[:,:,:2].reshape((data.fields.shape[0], n2_blocks, n1_blocks, 2))

    cell_mask = np.zeros((n2_blocks, n1_blocks), dtype=bool)
    for r in range(n2_blocks):
        cell_mask[r, (r%2)::2] = True   # pick up alternate blocks (square 1's)
    rows, cols = np.where(cell_mask)

    cell_disps = np.full((len(frames), n2_blocks, n1_blocks, 2), np.nan)

    print(block_disps.shape, cell_disps.shape, cell_mask.shape)

    p = -1
    for frame in frames:
        p += 1
        for r, c in zip(rows, cols):
            if r==0 or r==n2_blocks-1 or c==0 or c==n1_blocks-1:
                continue
            cell_disps[p,r,c] = ( block_disps[frame,r,c]/2 + 
                (block_disps[frame,r,c+1] + block_disps[frame,r,c-1] +
                block_disps[frame,r+1,c] + block_disps[frame,r-1,c])/8 )

    return cell_disps, cell_mask


# Calculate strain tensor for each cell from unit cell displacements
def unitcell_strain(cell_disps: np.ndarray, cell_mask: np.ndarray, block_spacing = 15.):
    
    """ 
    Calculate local strain tensor for each cell (r,c) as
    e[i,j] = [du_1, du_2].[l1, l2]^(-1)
    du_1 is relative displacement of neighbouring cells along lattice vector l1
    du_2 is relative displacement of neighbouring cells along lattice vector l2
    
    Inputs: 
        cell_disps[frames,row,col,:] holds displacements of cell located at each square 1 [row, col],
                and holds nan values at each square 2 [row ± 1, col ± 1]
        block_spacing defines the reference distances between adjacent squares

    Outputs: 
        dilation d[i,j] = Trace(e[i,j])
        shear1 s1[i,j] = Trace(e[i,j].PauliZ)
        shear2 s2[i,j] = Trace(e[i,j].PauliX)
        strain tensor e
    """

    frames = range(len(cell_disps))         # mode range
    l1 = block_spacing * np.array([1,1])     # lattice vector 1
    l2 = block_spacing * np.array([-1,1])    # lattice vector 2
    n2_blocks = cell_disps.shape[1]
    n1_blocks = cell_disps.shape[2]

    rows, cols = np.where(cell_mask)

    delta_u = np.full((len(frames), n2_blocks, n1_blocks, 2, 2), np.nan)  
    # 2x2 matrix of relative displacements of neighbours of cell (i,j) along the two lattice vectors
    # This array will have many extra entries (NaN values)

    strain = np.full((len(frames), n2_blocks, n1_blocks, 2, 2), np.nan)

    delta_r = np.array([l1,l2]).T            # 2x2 matrix with lattice vectors as columns
    delta_rinv = np.linalg.inv(delta_r)      # inverse of [l1,l2]

    # Calculating du
    for frame in frames:
        for r, c in zip(rows, cols):
            if r <= 1 or r >= n2_blocks-2 or c <= 1 or c >= n1_blocks-2:
                continue
            # du is relative displacement of neighbours along lattice vectors
            delta_u[frame,r,c] = np.array([cell_disps[frame,r+1,c+1] - cell_disps[frame,r-1,c-1],cell_disps[frame,r+1,c-1] - cell_disps[frame,r-1,c+1]]).T

    # Calculating strain
    for frame in frames:
        for r, c in zip(rows, cols):
            if r <= 1 or r >= n2_blocks-2 or c <= 1 or c >= n1_blocks-2:
                continue
            # strain is calculated as du.[l1,l2]^(-1) / 2
            strain[frame,r,c] = np.dot(delta_u[frame,r,c], delta_rinv)/2
    
    d = np.trace(strain, axis1=3, axis2=4)
    s1 = np.trace(np.dot(strain, np.array([[1,0],[0,-1]])), axis1=3, axis2=4)
    s2 = np.trace(np.dot(strain, np.array([[0,1],[1,0]])), axis1=3, axis2=4)
    stot = np.sqrt(np.square(s1)+np.square(s2))
    r = np.trace(np.dot(strain, np.array([[0,-1],[1,0]])), axis1=3, axis2=4)
    
    return strain, d, s1, s2, stot, r

# Elastic moduli for RS metamaterial
def RSmoduli(initial_angle, spacing, k_rot, k_stretch, k_shear, l):
    if k_shear is None:
        k_shear = k_stretch
    if l is None:
        l = 0.
        
    B = 2*k_stretch*k_rot/(4*k_rot + k_stretch * (spacing-l)**2 * np.tan(initial_angle)**2)
    G1 = k_stretch/2
    G2 = k_shear/2
    M = k_rot/4 + (k_shear/16) * spacing**2 + (k_stretch/16) * spacing**2 * (np.tan(initial_angle)**2 + (16*k_rot**2 + k_stretch * spacing**2 * np.tan(initial_angle)**2 * (4*k_rot + k_shear * spacing**2)) / ((4*k_rot + k_stretch * spacing**2 * np.tan(initial_angle)**2)**2))

    return B, G1, G2, M


# ==============================================================================
# COMPLEX ANALYSIS FUNCTIONS
# ==============================================================================

# Convert between complex and vector notation
def vtoz(vs):
    return vs[..., 0] + 1j * vs[..., 1]

def ztov(zs):
    return np.stack([zs.real, zs.imag], axis=-1)

# Construct complex analytic polynomial from coefficients
def polyconf(coeffs, n = None, L = 1.):
    if n is None:
        n = coeffs.shape[-1]
    def u_fit(z):
        return sum([coeffs[i] * (z/L)**i for i in range(n)])
    return u_fit

# Calculate error-in-fitting of displacement data for a given fitting function
def ERRfit(data, u_fit, zs, frac = False):
    """Error in a given fitting conformal map:
    Δ^2 = [Sum over all blocks of |udata[i] - ufit(z[i])|^2] 
    If frac = True, divide by [Sum over all blocks of |u_data[i]|^2]
    Else return absolute error without normalization
    """

    diff_sq = sum([abs(data[i]-u_fit(zs[i]))**2 for i in range(data.shape[0])])
    norm_sq = sum([abs(data[i])**2 for i in range(data.shape[0])]) if frac else 1
    return diff_sq/norm_sq

# CONFOMAL FITTING OF DATA
def conformal_fit(zs, L, u : np.ndarray, nc, err_calc = False, frac = False, ignore_translation = False):
    ''' INPUTS: zs - block centroid positions
                L - length scale (helps keep computations manageably small)
                u - displacement data in complex notation (no. of modes/timepoints, no. of blocks)
                nc - no. of terms in conformal fitting polynomial
                frac - if True calculate fractional error, else calculate absolute error
        OUTPUTS: for best fitting analytic functions of order nc for each displacement field,
                coeff_arr - array of coefficients of z^n
                u_fits - array of fitting functions that take z as input
                error_sq - array of squared error in each fit
    '''

    if ignore_translation:
        for m in range(u.shape[0]):
            u[m] = u[m] - np.mean(u[m])

    C = np.zeros((u.shape[0],nc), dtype=complex)
    A = np.array([[np.dot((zs/L)**i, (np.conjugate(zs)/L)**j) for i in range(nc)] for j in range(nc)])
    for m in range(u.shape[0]):
        index = np.isfinite(u[m])
        if np.prod(index)==0:
            print("Encountered (and ignored) NaN or Inf values at index :", f'{m}, {np.where(index==0)}')
        B = np.array([np.dot(u[m,index], (np.conjugate(zs[index])/L)**i) for i in range(nc)])
        C[m] = scipy.linalg.solve(
                A, B
            )
    coeff_arr = C
    u_fits = np.array([polyconf(coeff_arr[m,:], nc, L) for m in range(u.shape[0])])

    if err_calc:
        error_sq = np.empty(u.shape[0])
        for i in range(u.shape[0]):
            index = np.isfinite(u[i])
            error_sq[i] = ERRfit(u[i,index], u_fits[i], zs[index], frac)
    else: error_sq = None

    return coeff_arr, u_fits, error_sq


# CONSTRUCT MASK TO REMOVE BOUNDARY BLOCKS FROM DATA ANALYSIS
def remove_layers(n1_blocks, n2_blocks, n = 1):
    if n >= n2_blocks/2:
        raise Exception("Too many layers being removed!")
    layer_mask = np.ones(n1_blocks*n2_blocks, dtype = bool)
    if n==0:
        return layer_mask
    layer_mask[0:n*n1_blocks] = 0
    layer_mask[-n*n1_blocks:] = 0
    for j in range(n,n2_blocks-n):    
        layer_mask[j*n1_blocks:j*n1_blocks+n] = 0
        layer_mask[(j+1)*n1_blocks-n:(j+1)*n1_blocks] = 0
    return layer_mask



# ==============================================================================
# DEFINE FUNCTIONS FOR CONSERVATION ANALYSIS
# ==============================================================================

# Analytical Estimate for Proportionality Constant between Dilation and Bending Field (theta_co = -alpha d)
def calculate_alpha(
    spacing: float,
    hinge_length: float,
    initial_angle: float,
    mechanical_params: Optional[MechanicalParams] = None,
    k_stretch: Optional[Union[float, np.ndarray]] = None,
    k_rot: Optional[Union[float, np.ndarray]] = None,
) -> np.ndarray:
    """  
    Alpha relates the bending field with the coarse-grained dilation strain in the linear low-frequency limit

    Args:
        spacing (float): Spacing parameter of the geometry.
        hinge_length (float): Length of the hinge (ligament).
        initial_angle (float): Initial angle in radians.
        mechanical_params (MechanicalParams, optional): Mechanical parameters containing bond_params
            with k_stretch and k_rot stiffnesses. If provided, k_stretch and k_rot parameters
            will override the values from mechanical_params if also specified.
        k_stretch (float or np.ndarray, optional): Stretching stiffness. Can be scalar or 1D array.
            If not provided, extracted from mechanical_params.
        k_rot (float or np.ndarray, optional): Rotational stiffness. Can be scalar or 1D array.
            If not provided, extracted from mechanical_params.
    
    Returns:
        np.ndarray: Alpha value(s). If k_rot is a scalar, returns shape (1,).
                    If k_rot is an array, returns shape (n_k_rot,).
    
    Raises:
        ValueError: If neither mechanical_params nor both k_stretch and k_rot are provided.
    """
    # Extract k_stretch and k_rot from mechanical_params if not provided directly
    if k_stretch is None or k_rot is None:
        if mechanical_params is None:
            raise ValueError(
                "Either mechanical_params must be provided, or both k_stretch and k_rot "
                "must be specified directly."
            )
        bond_params = mechanical_params.bond_params
        if k_stretch is None:
            k_stretch = bond_params.k_stretch
        if k_rot is None:
            k_rot = bond_params.k_rot
    
    # Convert to numpy arrays for consistent handling
    k_rot_array = np.atleast_1d(np.asarray(k_rot, dtype=float))
    k_stretch_array = np.atleast_1d(np.asarray(k_stretch, dtype=float))
    
    # Handle broadcasting: if both are 1D arrays, they must have compatible shapes
    # (either same length, or one is length 1)
    if k_rot_array.ndim > 1 or k_stretch_array.ndim > 1:
        raise ValueError("k_rot and k_stretch must be scalar or 1D arrays")
    
    # Broadcast to same length if needed
    if len(k_rot_array) == 1 and len(k_stretch_array) > 1:
        k_rot_array = np.broadcast_to(k_rot_array, k_stretch_array.shape)
    elif len(k_stretch_array) == 1 and len(k_rot_array) > 1:
        k_stretch_array = np.broadcast_to(k_stretch_array, k_rot_array.shape)
    elif len(k_rot_array) != len(k_stretch_array):
        raise ValueError(
            f"k_rot and k_stretch must have compatible shapes. "
            f"Got k_rot: {k_rot_array.shape}, k_stretch: {k_stretch_array.shape}"
        )
    
    # Compute intermediate terms
    spacing_diff = spacing - hinge_length
    tan_angle = np.tan(initial_angle)
    
    numerator = k_stretch_array * spacing * spacing_diff * tan_angle
    denominator = 8 * k_rot_array + 2 * k_stretch_array * spacing_diff**2 * tan_angle**2
    
    alpha = numerator / denominator
    
    return alpha


# Convert complex displacement function f(z, zbar) to discrete displacement and dilation/rotation fields at block centroids
def apply_lambdify_to_complex_coords(func, coords, alpha, n1_blocks):
    """
    Apply a lambdified function f(z, z̄) to an array of (x, y) coordinates by converting to complex coordinates.
    Attach the estimated rotation field based on coarse-grained dilation and rotation fields using the proportionality constant alpha.

    Parameters:
    - func: A lambdified function that accepts two complex inputs (z, z̄).
    - coords: A NumPy array of shape (N, 2), where each row is an (x, y) pair.
    - alpha: The proportionality constant between dilation and bending fields.
    - n1_blocks: Number of blocks in the first direction.

    Returns:
    - ufield: array of shape (N, 3) containing the displacement field (ux, uy)
            and the estimated rotation field (theta_co + (-1)^i * (-1)^(i//n1_blocks) * theta_d).
    - diln: array of shape (N,) containing the dilation field at each coordinate.
    - rotn: array of shape (N,) containing the rotation field at each coordinate.
    """
    coords = np.asarray(coords)
    x_vals, y_vals = coords[:, 0], coords[:, 1]
    z_vals = x_vals + 1j * y_vals  # z = x + I*y
    zbar_vals = x_vals - 1j * y_vals  # z̄ = x - I*y

    z, zb = symbols('z zb')
    holo_strain = lambdify((z, zb), diff(func(z,zb),z))

    diln = 2*np.real(holo_strain(z_vals, zbar_vals)) + np.zeros(z_vals.shape)
    rotn = 2*np.imag(holo_strain(z_vals, zbar_vals)) + np.zeros(z_vals.shape)
    theta_d = -alpha * diln
    theta_co = rotn/2

    ufield = np.zeros((z_vals.shape[0], 3))
    ufield[:,:2] = ztov(func(z_vals, zbar_vals))
    ufield[:,2] = np.array([theta_co[i] + (-1)**(i) * (-1)**(i//n1_blocks) * theta_d[i] for i in range(z_vals.shape[0])])

    return ufield, diln, rotn

# PRECOMPUTE TOOLS FOR CONSERVATION ANALYSIS
def tools_for_conservation_analysis(geometry, initial_angle, k_stretch, k_rot, spacing, hinge_length, density):
    """
    Setup all the tools needed for conservation analysis.
    Returns a dictionary with all precomputed quantities (vectorized over k_rot).
    Args:
        geometry: RotatedSquareGeometry object
        initial_angle: Initial angle in radians
        k_stretch: Stiffness parameter (scalar)
        k_rot: Stiffness parameter (scalar or array)
        spacing, hinge_length: Geometric parameters
        density: Material density
    Returns:
        tools: dict with keys:
            - 'map_fields': List of field maps for projection
            - 'map_norm': Normalization factors (shape: (n_maps,) or (n_k_rot, n_maps))
            - 'inertia': Mass/inertia matrix
            - 'bcs': Reference block centroids (centered)
            - 'alpha': Geometric parameter (scalar or array of shape (n_k_rot,))
            - 'funcs': Lambdified functions for map expressions
            - 'mono_exprs': Original symbolic expressions for the maps
            - 'k_rot': The k_rot values used (shape: (n_k_rot,))
    """

    block_centroids, centroid_node_vectors, _, _ = geometry.get_parametrization()

    _inertia = compute_inertia(
        vertices=centroid_node_vectors(initial_angle),
        density=density).reshape((geometry.n_blocks * 3,))
    
    # SHIFT ORIGIN TO CENTRE OF LATTICE
    bcs_raw = block_centroids(initial_angle)
    origin = np.array([bcs_raw[0, 0] + bcs_raw[-1, 0], bcs_raw[0, 1] + bcs_raw[-1, 1]])/2
    bcs = bcs_raw - origin

    k_rot_array = np.atleast_1d(k_rot)
    alpha = calculate_alpha(spacing, hinge_length=hinge_length, initial_angle=initial_angle, 
                       k_stretch=k_stretch, k_rot=k_rot_array)
    
    z, zb = symbols('z zb')
    L = 1                       # scaling length
    
    mono_exprs = [
            z/L, zb/L, I*zb/L,
            z**2/L**2, I*z**2/L**2,
            z*zb/L**2, I*z*zb/L**2,
            zb**2/L**2, I*zb**2/L**2,
            I*z/L, 1, I,
        ]
    
    funcs = [lambdify((z, zb), expr) for expr in mono_exprs]

    if len(k_rot_array) == 1:
        map_fields = [apply_lambdify_to_complex_coords(f, bcs, alpha[0], geometry.n1_blocks)[0] for f in funcs]
        map_norm = np.sqrt(np.array([fields.flatten() @ (_inertia * fields.flatten()) for fields in map_fields]))
    else:
        map_fields = [[apply_lambdify_to_complex_coords(f, bcs, a, geometry.n1_blocks)[0] for f in funcs] for a in alpha]
        map_norm = np.sqrt(np.array([[fields.flatten() @ (_inertia * fields.flatten()) for fields in mf] for mf in map_fields]))
        
    return {
        'map_fields': map_fields,
        'map_norm': map_norm,
        'inertia': _inertia,
        'bcs': bcs,
        'alpha': alpha,
        'funcs': funcs,
        'mono_exprs': mono_exprs,
        'k_rot': k_rot_array,
    }

# If k_rot is vectorized, this function allows us to slice the tools dictionary to get the corresponding map fields, norms, and alpha for a specific index.
def slice_tools_for_param(tools, idx):
    """Return a single-parameter view of tools when k_rot is vectorized.
    If only one k_rot is present, returns the original tools.
    """
    k_rot_arr = np.atleast_1d(tools['k_rot'])
    if k_rot_arr.shape[0] == 1:
        return tools
    return {
        'map_fields': tools['map_fields'][idx],
        'map_norm': tools['map_norm'][idx],
        'inertia': tools['inertia'],
        'bcs': tools['bcs'],
        'alpha': np.atleast_1d(tools['alpha'][idx]),
        'funcs': tools['funcs'],
        'mono_exprs': tools['mono_exprs'],
        'k_rot': np.atleast_1d(k_rot_arr[idx]),
    }


# PROJECTOR FUNCTIONS
def component_along_map(response_field : np.ndarray, map_field : np.ndarray,):
    return np.einsum('ij,j->i', response_field, map_field.flatten())

def projection_analysis(response_data, tools):
    """
    Generic projector: map any provided fields onto precomputed map_fields.
    No physics-specific assumptions; returns projections for every supplied entry.

    Args:
        response_data: dict or array
            - array -> interpreted as {'field': array}
            - dict  -> each value can be an array (n_time, n_dof) or list of such arrays
        examples of response_data types:
            - {'momentum': np.ndarray or list, 'momentum_rate': np.ndarray or list, 'ext_forces': np.ndarray or list}
            - np.ndarray or list (single field)
            (each entry must have shape (n_timepoints, n_dof) or be a 1D list of such arrays)

        tools: dict from setup_analysis_tools()

    Returns:
        results: dict containing 'map_norm' and 'map_<key>' for each input key.
    """

    map_fields = tools['map_fields']
    projected_results = {'map_norm': tools['map_norm']}

    if isinstance(response_data, np.ndarray):
        response_data = {'field': response_data}
    elif not isinstance(response_data, dict):
        raise ValueError("response_data must be a dict or numpy array")
    
    def projected_value(value):
        if isinstance(value, list):
            return [np.array([component_along_map(arr, field) for field in map_fields]) for arr in value]
        return np.array([component_along_map(value, field) for field in map_fields])
    
    for key, value in response_data.items():
        projected_results[f'map_{key}'] = projected_value(value)
    return projected_results


# PROCESS TIMESERIES DATASET TO COMPUTE MOMENTUM, MOMENTUM RATE, AND EXTERNAL FORCES
def process_dataset(tpts, fields, inertia, forces=None, stiffness=None, damping=None, fields_rate=None):
    '''Process timeseries dataset to compute momentum, momentum rate, and external forces.'''

    # Extract displacement and velocity fields
    disps = fields[:,0].reshape(fields.shape[0], -1)
    velocities = fields[:,1].reshape(fields.shape[0], -1)

    # Compute momentum
    momentum = velocities * inertia[np.newaxis, :]

    if fields_rate is None:
        momentumrate = np.gradient(momentum, tpts, axis=0)          # time derivative
    else:
        accelerations = fields_rate[:,1].reshape(fields_rate.shape[0], -1)
        momentumrate = accelerations * inertia[np.newaxis, :]

    # Compute external forces if not provided
    if forces is None:
        if stiffness is None or damping is None:
            raise ValueError("Stiffness and damping coefficients must be provided if forces are not given.")
        ext_forces = momentumrate - (stiffness @ disps.T).T - (damping @ velocities.T).T
    else:
        ext_forces = forces.reshape(fields.shape[0], -1)

    return {
        'momentum': momentum,
        'momentum_rate': momentumrate,
        'ext_forces': ext_forces,
    }

# COMPLETE PIPELINE: LOAD DATASET -> COMPUTE DYNAMICS -> PROJECT ONTO MAPS -> RETURN RESULTS
def analyse_dataset_conservation(tpts, fields, tools, forces=None, stiffness=None, damping=None, fields_rate = None):
    """
    Complete pipeline: load dataset -> compute dynamics -> project onto maps -> return results.
    Memory-efficient: processes one dataset at a time.
    
    Args:
        dataset_info: dict or str path to dataset
        tools: dict from setup_analysis_tools()
        stiffness, damping: optional matrices for force computation
    
    Returns:
        dict with 'map_momentum', 'map_momentum_rate', 'map_ext_forces', etc.
    """
    # Process
    data = process_dataset(
        tpts,
        fields=fields,
        forces=forces,
        inertia=tools['inertia'],
        stiffness=stiffness,
        damping=damping,
        fields_rate=fields_rate,
    )
    
    # Project onto maps using the precomputed map fields
    results = projection_analysis(data, tools)
    
    # Add timepoints to results
    results['timepoints'] = tpts
    
    return results





# ==============================================================================
# PLOTTING MODE-PROFILE FUNCTIONS (SLIGHTLY MODIFIED VERSIONS OF FUNCTIONS IN blockymetamaterials.utils)
# ==============================================================================

from pathlib import Path

from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from blockymetamaterials.geometry import rotation_matrix
from blockymetamaterials.utils import EigenmodeData

def orange_blue_cmap():
    """
    Custom colormap
    """

    top = mpl.colormaps['Oranges_r'].resampled(128)
    bottom = mpl.colormaps['Blues'].resampled(128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    return ListedColormap(newcolors, name='OrangeBlue')

def prepare_mode_figure(data: EigenmodeData, field, mode_range, figsize, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True):

    if field == "ux":
        field_values = data.fields[:, :, 0]
        _legend_label = r"$u_1$"
    elif field == "uy":
        field_values = data.fields[:, :, 1]
        _legend_label = r"$u_2$"
    elif field == "theta":
        field_values = data.fields[:, :, 2]
        _legend_label = r"$\theta$"
    elif field == "u":
        field_values = (data.fields[:, :, 0]**2 + data.fields[:, :, 1]**2)**0.5
        _legend_label = r"$\displaystyle\frac{\vert \mathbf{u} \vert}{u_{\mathrm{max}}}$"
    elif field == "theta_abs":
        field_values = np.abs(data.fields[:, :, 2])
        _legend_label = r"$\lvert\theta\rvert$"
    else:
        raise ValueError

    vmin, vmax = vlim if vlim is not None else (None, None)
    _legend_label = legend_label if legend_label is not None else _legend_label

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    axes.axis("equal")
    axes.tick_params(labelsize=ticksize)
    if not axis:
        axes.set_axis_off()
    cb = fig.colorbar(
        cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
        ax=axes,
        pad=0.02,
        label=_legend_label,
        aspect=40
    )
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_ylabel(_legend_label, fontsize=fontsize, rotation=0, labelpad=15, va='center')
    frames = range(len(data.fields)) if mode_range is None else mode_range

    return field_values, fig, axes, frames


def generate_polygons(block_centroids, centroid_node_vectors, block_displacements=None, deformed=False):
    """
    docstring
    """

    if deformed and block_displacements is not None:
        polygons = [
            Polygon((rotation_matrix(DOFs[-1]) @ vertices.T).T + centroid + DOFs[:2])
            for vertices, centroid, DOFs in zip(centroid_node_vectors, block_centroids, block_displacements)]
    else:
        polygons = [Polygon(vertices + centroid, closed=True)
                    for vertices, centroid in zip(centroid_node_vectors, block_centroids)]

    return polygons

def generate_patch_collection(block_centroids, centroid_node_vectors, block_displacements=None, field_values=None, deformed=False, clim=None, cmap=orange_blue_cmap(), alpha=0.95):
    """
    docstring
    """

    polygons = generate_polygons(block_centroids, centroid_node_vectors,
                                 block_displacements=block_displacements, deformed=deformed)
    patches = PatchCollection(polygons, cmap=cmap, alpha=alpha)
    if field_values is not None:
        patches.set_array(field_values)
        min_value, max_value = (field_values.min(), field_values.max()) if clim is None else clim
        patches.set_clim(min_value, max_value)
    patches.set(edgecolor="black", linewidth=1)

    return patches

def generate_mode_images(data: EigenmodeData, field, out_dir, deformed=False, mode_range=None, scale_deformation=1, figsize=None, xlim=None, ylim=None, dpi=200, geometry=None, mesh=None, cmap=orange_blue_cmap(), vlim=None, legend_label=None, fontsize=14, ticksize=14, axis=True, save_label = None,
    lattice = True, latt_alpha = 0.95,  var = r'\omega', meshcol = 'k'):
    """
    mesh=None: if set to True, a mesh connecting the centroids of each block is superimposed on the images
    docstring
    """

    field_values, fig, axes, frames = prepare_mode_figure(
        data, field, mode_range, figsize, cmap=cmap, vlim=vlim, legend_label=legend_label, fontsize=fontsize, ticksize=ticksize, axis=axis
    )
    block_centroids = data.block_centroids
    centroid_node_vectors = data.centroid_node_vectors
    block_displacements = data.fields

    flag = 0

    for i in frames:
        # Each frame refer to a mode
        axes.clear()
        if lattice:
            patches = generate_patch_collection(
                block_centroids=block_centroids,
                centroid_node_vectors=centroid_node_vectors,
                block_displacements=block_displacements[i, :, :] * scale_deformation,
                field_values=field_values[i],
                deformed=deformed,
                clim=None,  # Normalize colors between min and max
                cmap=cmap,
                alpha=latt_alpha
            )
            axes.add_collection(patches)
        if var==r'f':
            axes.set_title(fr"${var}_{{{i+1}}}={data.eigenvalues[i]:.2f}$ Hz", fontsize=fontsize)
        elif var=='map':
            axes.set_title(fr"map : {data.eigenvalues[i]}", fontsize=fontsize)
        elif var==r'$\omega$':
            axes.set_title(fr"$\omega_{{{i+1}}}={data.eigenvalues[i]:.2f}$ rad/s", fontsize=fontsize)
        else:
            axes.set_title(fr"${var}_{{{i+1}}}={data.eigenvalues[i]:.2f}$", fontsize=fontsize)
        axes.set(xlim=xlim, ylim=ylim)

        if mesh == True:
            n1 = geometry.n1_blocks
            n2 = geometry.n2_blocks
            for j in np.arange(geometry.n2_blocks):
                row_block_coordinates = np.array([block_centroids[n1*j:n1*(j+1), 0] + block_displacements[i, n1*j:n1*(j+1), 0]*scale_deformation,
                                                  block_centroids[n1*j:n1*(j+1), 1] + block_displacements[i, n1*j:n1*(j+1), 1]*scale_deformation])
                axes.plot(row_block_coordinates[0, :], row_block_coordinates[1, :], meshcol)

            for k in np.arange(geometry.n1_blocks):
                col_block_coordinates = np.array([block_centroids[k:n1*(n2-1)+k+1:n1, 0] + block_displacements[i, k:n1*(n2-1)+k+1:n1, 0]*scale_deformation,
                                                  block_centroids[k:n1*(n2-1)+k+1:n1, 1] + block_displacements[i, k:n1*(n2-1)+k+1:n1, 1]*scale_deformation])
                axes.plot(col_block_coordinates[0, :], col_block_coordinates[1, :], meshcol)
        if not axis:
            axes.set_axis_off()
        if save_label is None:
            out_path = Path(f"{str(out_dir)}/{i:04d}.pdf")
        else:
            out_path = Path(f"{str(out_dir)}/{save_label[flag]}.pdf")
            flag = flag+1
        out_path.parent.mkdir(parents=True, exist_ok=True)  # Make sure parents directories exist
        fig.savefig(str(out_path), dpi=dpi)

    plt.close(fig)