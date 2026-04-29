"""
The `loading` module implements loading and boundary conditions data structures.
"""

from typing import Callable, Dict

from jax import grad, vmap
import jax.numpy as jnp

from blockymetamaterials.energy import vdot
from blockymetamaterials.geometry import DOFsInfo, Geometry, rotation_matrix
from blockymetamaterials.utils import ControlParams, is_scalar
from blockymetamaterials.kinematics import block_to_node_kinematics, build_constrained_kinematics


def build_loading(
        geometry: Geometry,
        loaded_block_DOF_pairs: jnp.ndarray,
        loading_fn: Callable,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([])):
    """Defines the loading function.

    Args:
        geometry (Geometry): geometry.
        loaded_block_DOF_pairs (jnp.ndarray): array of shape (Any, 2) where each row defines a pair of [block_id, DOF_id] where DOF_id is either 0, 1, or 2
        loading_fn (Callable): Loading function. Output shape should either be scalar or match (len(loaded_block_DOF_pairs),).
        constrained_block_DOF_pairs (jnp.ndarray, optional): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        Callable: vector loading function evaluating to `loading_fn` for the DOFs defined by `loaded_block_DOF_pairs` and 0 otherwise.
    """

    # loaded DOF ids based on global numeration
    loaded_DOF_ids = jnp.array([block_id * 3 + DOF_id for block_id, DOF_id in loaded_block_DOF_pairs])
    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, _, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    def global_loading_fn(state, t, loading_params: Dict):

        loading_vector = jnp.zeros((len(all_DOF_ids),))
        loading_vector = loading_vector.at[loaded_DOF_ids].set(
            loading_fn(state, t, **loading_params)
        )
        loading_vector = loading_vector[free_DOF_ids]  # Reduce loading vector to the free DOFs

        return loading_vector

    return global_loading_fn


def build_node_loading(
        geometry: Geometry,
        loaded_block_node_DOF_triples: jnp.ndarray,
        loading_fn: Callable,
        centroid_node_vectors: jnp.ndarray,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([])):
    """
    docstring
    """

    # TODO: Implement nodal loading function in one of the following ways:
    #   - Compute virtual power and let jax take the gradient with respect to virtual velocity.
    #   - Find the appropriate way to vectorize something like (A_n)^T . F_n where A_n is the gradient of n node displacement with respect to block DOFs and F_n the nodal loading.
    # In both cases, be sure to constrained the resulting loading vector to the freeDOFs using constraints info.

    # node_displacements = block_to_node_kinematics(
    #     block_displacement,
    #     centroid_node_vectors
    # )


def build_viscous_damping(
        geometry: Geometry,
        damped_blocks: jnp.ndarray,
        constrained_block_DOF_pairs: jnp.ndarray = jnp.array([])):
    """Defines viscous damping forces.

    Args:
        geometry (Geometry): geometry.
        damped_blocks (jnp.ndarray): array of shape (n_damped_blocks,) collecting the block ids of the damped blocks.
        damping_values (jnp.ndarray): array of shape (n_damped_blocks, 3) collecting the damping values for each block and DOF.
        constrained_block_DOF_pairs (jnp.ndarray): Array of shape (n_constraints, 2) where each row is of the form [block_id, DOF_id]. Defaults to jnp.array([]).

    Returns:
        Callable: function evaluating the viscous damping forces.
    """

    damped_DOF_ids = jnp.concatenate([jnp.arange(block_id * 3, (block_id + 1) * 3) for block_id in damped_blocks])
    # Retrieve free DOFs from constraints info (this information is assumed to be static)
    free_DOF_ids, _, all_DOF_ids = DOFsInfo(geometry.n_blocks, constrained_block_DOF_pairs)

    # This is to ensure correct shape of loading vector when damping is either a scalar or an array of shape (n_damped_blocks, 3)
    reshaping_array = jnp.ones((len(damped_blocks), 3))

    def loading_fn(state, t, damping: jnp.ndarray):
        _, velocity = state
        loading_vector = jnp.zeros((len(all_DOF_ids),))
        loading_vector = loading_vector.at[damped_DOF_ids].set(
            (damping * reshaping_array).reshape(damped_DOF_ids.shape)
        )
        loading_vector = loading_vector[free_DOF_ids]

        return -loading_vector * velocity

    return loading_fn


def build_bond_viscous_damping(
        geometry: Geometry,
        kinematics_fn: Callable = None,
        kinematics_dot_fn: Callable = None,):
    """Defines damping forces due to viscosity in the bonds."""

    bond_connectivity = geometry.bond_connectivity()

    def viscous_potential_fn(nodal_DOFs, nodal_DOFs_dot, internal_damping, reference_vector):
        DOFs1 = nodal_DOFs[bond_connectivity[:, 0]]
        DOFs2 = nodal_DOFs[bond_connectivity[:, 1]]
        DOFs1_dot = nodal_DOFs_dot[bond_connectivity[:, 0]]
        DOFs2_dot = nodal_DOFs_dot[bond_connectivity[:, 1]]
        dU = DOFs2[:, :2] - DOFs1[:, :2]
        bond_reference_length = vdot(reference_vector, reference_vector)**0.5
        bond_current_length = vdot(dU + reference_vector, dU + reference_vector)**0.5
        bond_current_direction = ((dU + reference_vector).T / bond_current_length).T
        axial_strain_rate = jnp.sum(bond_current_direction * (DOFs2_dot[:, :2] - DOFs1_dot[:, :2]), axis=-1)
        shear_strain_rate = (
            jnp.cross(
                bond_current_direction,
                (DOFs2_dot[:, :2] - DOFs1_dot[:, :2]),
                axis=-1,
            )/bond_current_length - (DOFs2_dot[:, 2] + DOFs1_dot[:, 2])/2
        )*bond_reference_length
        bending_strain_rate = DOFs2_dot[:, 2] - DOFs1_dot[:, 2]
        internal_damping = internal_damping * jnp.ones((bond_connectivity.shape[0], 3))
        return jnp.sum(
            0.5 * (
                internal_damping[:, 0]*axial_strain_rate**2 +
                internal_damping[:, 1]*shear_strain_rate**2 +
                internal_damping[:, 2]*bending_strain_rate**2
            )
        )

    # NOTE: For some reason this is faster than using jax.jacobian on the kinematics function.
    block_to_node_kinematics_velocity = vmap(
        vmap(
            lambda block_velocity, centroid_node_vector: block_velocity +
            block_velocity[2]*jnp.array([-centroid_node_vector[1], centroid_node_vector[0], 0]),
            in_axes=(None, 0)
        ),
        in_axes=(0, 0)
    )

    ############ original #############
    # def block_viscous_potential_fn(block_DOFs, block_DOFs_dot, control_params: ControlParams):
    #     nodal_DOFs = block_to_node_kinematics(
    #         block_DOFs, control_params.geometrical_params.centroid_node_vectors
    #     ).reshape(geometry.n_nodes, 3)
    #     nodal_DOFs_dot = block_to_node_kinematics_velocity(
    #         block_DOFs_dot, control_params.geometrical_params.centroid_node_vectors
    #     ).reshape(geometry.n_nodes, 3)
    #     return viscous_potential_fn(
    #         nodal_DOFs,
    #         nodal_DOFs_dot,
    #         control_params.mechanical_params.internal_damping,
    #         control_params.mechanical_params.bond_params.reference_vector
    #     )

    ########### incorrectly fixed ############
    # def block_viscous_potential_fn(block_DOFs, block_DOFs_dot, control_params: ControlParams):
    #     nodal_DOFs = block_to_node_kinematics(
    #         block_DOFs, control_params.geometrical_params.centroid_node_vectors
    #     )
    #     centroid_node_vectors_current = control_params.geometrical_params.centroid_node_vectors + nodal_DOFs[:, :, :2]
    #     nodal_DOFs_dot = block_to_node_kinematics_velocity(
    #     block_DOFs_dot, centroid_node_vectors_current
    #     )
    #     return viscous_potential_fn(
    #         nodal_DOFs.reshape(geometry.n_nodes, 3),
    #         nodal_DOFs_dot.reshape(geometry.n_nodes, 3),
    #         control_params.mechanical_params.internal_damping,
    #         control_params.mechanical_params.bond_params.reference_vector
    #     )

    ######### fixed ###############
    def current_centroid_node_vectors_fn(block_rotations, centroid_node_vectors):
        return vmap(lambda R, c: (rotation_matrix(R)@c.T).T, in_axes=(0, 0))(block_rotations, centroid_node_vectors)

    def block_viscous_potential_fn(block_DOFs, block_DOFs_dot, control_params: ControlParams):
        nodal_DOFs = block_to_node_kinematics(
            block_DOFs, control_params.geometrical_params.centroid_node_vectors
        )
        centroid_node_vectors_current = current_centroid_node_vectors_fn(
            block_DOFs[:, 2], control_params.geometrical_params.centroid_node_vectors
        )
        nodal_DOFs_dot = block_to_node_kinematics_velocity(
            block_DOFs_dot, centroid_node_vectors_current
        )
        return viscous_potential_fn(
            nodal_DOFs.reshape(geometry.n_nodes, 3),
            nodal_DOFs_dot.reshape(geometry.n_nodes, 3),
            control_params.mechanical_params.internal_damping,
            control_params.mechanical_params.bond_params.reference_vector
        )

    def block_viscous_potential_constrained_fn(free_DOFs, free_DOFs_dot, t, control_params: ControlParams):
        block_displacement = kinematics_fn(free_DOFs, t, control_params)
        block_velocity = kinematics_dot_fn(free_DOFs, free_DOFs_dot, t, control_params)
        return block_viscous_potential_fn(block_displacement, block_velocity, control_params)

    internal_viscous_force_fn = grad(
        lambda u, v, *args: -block_viscous_potential_constrained_fn(u, v, *args), argnums=1)

    def loading_fn(state, t, control_params: ControlParams):
        free_DOFs, free_DOFs_dot = state
        loading_vector = internal_viscous_force_fn(
            free_DOFs,
            free_DOFs_dot,
            t,
            control_params,
        )
        return loading_vector

    return loading_fn
