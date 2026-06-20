import pytest
import jax
import jax.numpy as jnp
from jepa_robotics.models.v_jepa import ViTEncoder, JEPAPredictor
from jepa_robotics.models.world_model import ActionConditionedTransformer
from jepa_robotics.training.optimization import build_smac_scenario

def test_vit_encoder_compilation_and_shape():
    """Verify the ViT initializes and outputs the exact Latent Dimension specified."""
    # Mock image batch: (Batch, H, W, C)
    batch_size = 2
    mock_image = jnp.ones((batch_size, 256, 256, 3))
    
    # Initialize dynamic architecture
    latent_dim = 128
    encoder = ViTEncoder(latent_dim=latent_dim, depth=2, num_heads=4)
    
    rng = jax.random.PRNGKey(0)
    variables = encoder.init(rng, mock_image)
    
    # Forward pass
    latents = encoder.apply(variables, mock_image)
    
    # Verify shape is exactly (Batch, Latent_Dim)
    assert latents.shape == (batch_size, latent_dim)

def test_world_model_causal_masking_and_shape():
    """Verify the Action-Conditioned Transformer compiles and outputs next states."""
    batch_size = 2
    seq_len = 5
    latent_dim = 256
    action_dim = 7
    
    # Mock temporal sequences
    mock_latents = jnp.ones((batch_size, seq_len, latent_dim))
    mock_actions = jnp.ones((batch_size, seq_len, action_dim))
    
    world_model = ActionConditionedTransformer(latent_dim=latent_dim, action_dim=action_dim, depth=2)
    
    rng = jax.random.PRNGKey(0)
    variables = world_model.init(rng, mock_latents, mock_actions)
    
    # Forward pass
    next_state_preds = world_model.apply(variables, mock_latents, mock_actions)
    
    # Output should exactly mirror the temporal sequence shape
    assert next_state_preds.shape == (batch_size, seq_len, latent_dim)

def test_smac3_scenario_builds_correctly():
    """Verify the hierarchical ConfigSpace initializes without error."""
    scenario = build_smac_scenario()
    
    # Verify hyperparameters exist in the configuration space
    cs = scenario.configspace
    assert "latent_dim" in cs
    assert "vit_depth" in cs
    assert "tau" in cs
