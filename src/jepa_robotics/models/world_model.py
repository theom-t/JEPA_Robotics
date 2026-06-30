import jax.numpy as jnp
from flax import linen as nn

class ActionConditionedTransformer(nn.Module):
    """
    Latent World Model that predicts the next latent state given a 
    historical sequence of latent states and 7D Cartesian actions.
    """
    latent_dim: int = 256
    action_dim: int = 7
    depth: int = 4
    num_heads: int = 8
    mlp_dim: int = 1024
    activation_fn: str = "gelu"
    
    @nn.compact
    def __call__(self, latents, actions):
        """
        latents: (batch_size, seq_len, latent_dim)
        actions: (batch_size, seq_len, action_dim)
        Returns predicted next states: (batch_size, seq_len, latent_dim)
        """
        # Ensure latents and actions have same sequence length
        b, seq_len, _ = latents.shape
        
        # Project actions into the same dimensional space as latents
        action_embeddings = nn.Dense(self.latent_dim)(actions)
        
        # Combine latents and actions (e.g., addition or concatenation)
        # Here we add them, which is a common approach in Decision Transformers
        combined = latents + action_embeddings
        
        # Add temporal positional embedding
        pos_embedding = self.param('temporal_pos_embedding', 
                                   nn.initializers.normal(stddev=0.02), 
                                   (1, seq_len, self.latent_dim))
        x = combined + pos_embedding
        
        # Create a causal mask to prevent attending to future states
        # Shape: (1, 1, seq_len, seq_len) for Flax SelfAttention
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = jnp.expand_dims(mask, axis=(0, 1))
        
        # Apply Transformer blocks
        for _ in range(self.depth):
            # Attention block
            y = nn.LayerNorm()(x)
            # Self-attention over time with explicit cuDNN FlashAttention backend
            y = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dot_product_attention_kwargs={"implementation": "cudnn"}
            )(y, y, mask=mask)
            x = x + y

            # MLP block
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.mlp_dim)(y)
            act = getattr(nn, self.activation_fn)
            y = act(y)
            y = nn.Dense(self.latent_dim)(y)
            x = x + y
            
        x = nn.LayerNorm()(x)
        
        # Predict the NEXT state (s_{t+1}) from the current token representation
        next_state_predictions = nn.Dense(self.latent_dim)(x)
        
        return next_state_predictions
