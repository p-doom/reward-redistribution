import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from stoix.utils.multistep import (
    batch_truncated_generalized_advantage_estimation,
)


class TruncatedGeneralizedAdvantageEstimationTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
    
    @chex.all_variants()
    def test_advantage_no_bootstrapping(self) -> None:
        """Test GAE advantage outputs without value bootstrapping in a sparse reward setting.

        We are using lambda=1 and discount (gamma)=1.
        """
        
        r_t = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        v_t = jnp.ones((2, 6))
        v_t = v_t.at[:, -1].set(0.0)

        discount_t = jnp.ones_like(r_t)
        truncation_t = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        
        advantage_fn_variant = self.variant(batch_truncated_generalized_advantage_estimation)
        advantages, targets = advantage_fn_variant(
            r_t=r_t, discount_t=discount_t, lambda_=1.0, values=v_t, truncation_flags=truncation_t
        )

        expected_advantages = -jnp.array([[1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]) 
        expected_targets = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]) 

        np.testing.assert_allclose(advantages, expected_advantages, atol=1e-3)
        np.testing.assert_allclose(targets, expected_targets, atol=1e-3)

    @chex.all_variants()
    def test_advantage_with_bootstrapping(self) -> None:
        """Test GAE advantage outputs with value bootstrapping in a sparse reward setting.

        We are using lambda=1 and discount (gamma)=1.
        """
        
        r_t = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        v_t = jnp.ones((2, 6))

        discount_t = 1.-r_t
        truncation_t = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        
        advantage_fn_variant = self.variant(batch_truncated_generalized_advantage_estimation)
        advantages, targets = advantage_fn_variant(
            r_t=r_t, discount_t=discount_t, lambda_=1.0, values=v_t, truncation_flags=truncation_t
        )

        expected_advantages = jnp.zeros_like(r_t)
        expected_targets = jnp.ones_like(r_t)

        np.testing.assert_allclose(advantages, expected_advantages, atol=1e-3)
        np.testing.assert_allclose(targets, expected_targets, atol=1e-3)

    @chex.all_variants()
    def test_no_bootstrapping(self) -> None:
        """Test GAE without value bootstrapping in a sparse reward setting.

        We are using lambda=1 and discount (gamma)=1.
        """
        
        r_t = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        # simulate a noisy/randomly initialized value function 
        v_t = jax.random.uniform(jax.random.PRNGKey(0), (2, 6))
        # set last_val to 0
        v_t = v_t.at[:, -1].set(0.0)

        discount_t = 1.-r_t
        truncation_t = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        
        advantage_fn_variant = self.variant(batch_truncated_generalized_advantage_estimation)
        _, targets = advantage_fn_variant(
            r_t=r_t, discount_t=discount_t, lambda_=1.0, values=v_t, truncation_flags=truncation_t
        )

        expected_targets = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
        np.testing.assert_allclose(targets, expected_targets, atol=1e-3)

    @chex.all_variants()
    def test_with_bootstrapping(self) -> None:
        """Test GAE with value bootstrapping in a sparse reward setting.

        We are using lambda=1 and discount (gamma)=1.
        """
        
        r_t = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        # simulate a noisy/randomly initialized value function 
        # simulate bootstrapping for the last_val
        v_t = jax.random.uniform(jax.random.PRNGKey(0), (2, 6))

        discount_t = 1.-r_t
        truncation_t = jnp.array([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        
        advantage_fn_variant = self.variant(batch_truncated_generalized_advantage_estimation)
        _, targets = advantage_fn_variant(
            r_t=r_t, discount_t=discount_t, lambda_=1.0, values=v_t,truncation_flags=truncation_t
        )

        expected_targets = jnp.ones_like(r_t)
        expected_targets = expected_targets.at[0].set(expected_targets[0] * v_t[0,-1])

        np.testing.assert_allclose(targets, expected_targets, atol=1e-3)



if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()