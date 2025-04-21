import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from stoix.base_types import Observation


class EmbeddingInput(nn.Module):
    """JAX Array Input."""

    @nn.compact
    def __call__(self, embedding: chex.Array) -> chex.Array:
        return embedding


class ObservationInput(nn.Module):
    """Only Observation Input."""

    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        observation = observation.agent_view
        return observation

class ObservationTimestepInput(nn.Module):
    """Observation and Timestep Input."""

    @nn.compact
    def __call__(self, observation: Observation, timestep: chex.Array) -> chex.Array:
        agent_view = observation.agent_view
        # Ensure timestep has the same batch dimension and an added feature dimension
        if timestep.ndim == 1:
            timestep = jnp.expand_dims(timestep, axis=-1)
        # Ensure timestep is broadcastable to the agent_view's shape
        # This assumes agent_view is (batch_size, features) or similar
        # and timestep is (batch_size, 1)
        # If agent_view has more dims (e.g., sequence), broadcasting might be needed differently
        # For simplicity, assuming typical MLP input shapes
        if agent_view.ndim > 2 and timestep.ndim == 2:
             # Example: agent_view is (batch, seq_len, features), timestep is (batch, 1)
             # We might want to broadcast timestep to (batch, seq_len, 1)
             timestep = jnp.expand_dims(timestep, axis=1)
             timestep = jnp.repeat(timestep, agent_view.shape[1], axis=1)


        x = jnp.concatenate([agent_view, timestep], axis=-1)
        return x

class ObservationActionInput(nn.Module):
    """Observation and Action Input."""

    @nn.compact
    def __call__(self, observation: Observation, action: chex.Array) -> chex.Array:
        observation = observation.agent_view
        x = jnp.concatenate([observation, action], axis=-1)
        return x


class EmbeddingActionInput(nn.Module):

    action_dim: int

    @nn.compact
    def __call__(self, observation_embedding: chex.Array, action: chex.Array) -> chex.Array:
        x = jnp.concatenate([observation_embedding, action], axis=-1)
        return x


class EmbeddingActionOnehotInput(nn.Module):

    action_dim: int

    @nn.compact
    def __call__(self, observation_embedding: chex.Array, action: chex.Array) -> chex.Array:
        action_one_hot = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([observation_embedding, action_one_hot], axis=-1)
        return x
