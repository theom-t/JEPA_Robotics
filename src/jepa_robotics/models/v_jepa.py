import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable

class MLPBlock(nn.Module):
    """A standard MLP block used in Vision Transformers."""
    mlp_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.mlp_dim)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim)(x)
        return x

class Encoder1DBlock(nn.Module):
    """Transformer block for the ViT encoder."""
    mlp_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, inputs):
        # Attention block
        x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(num_heads=self.num_heads)(x)
        x = x + inputs

        # MLP block
        y = nn.LayerNorm()(x)
        y = MLPBlock(mlp_dim=self.mlp_dim, out_dim=inputs.shape[-1])(y)
        return x + y

class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) acting as the Context (E_x) or Target (E_y) Encoder.
    Compresses image patches into latent representations.
    """
    latent_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_dim: int = 1024
    patch_size: int = 16

    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, H, W, C)
        b, h, w, c = x.shape
        
        # Simple patch embedding: we use a dense layer on flattened patches
        # For a more robust implementation, we would use a Conv layer with stride=patch_size
        x = nn.Conv(features=self.latent_dim, kernel_size=(self.patch_size, self.patch_size), 
                    strides=(self.patch_size, self.patch_size), padding="VALID")(x)
        
        # Flatten spatial dimensions
        x = x.reshape((b, -1, self.latent_dim))
        
        # Add positional embedding (learned)
        num_patches = x.shape[1]
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, num_patches, self.latent_dim))
        x = x + pos_embedding

        # Apply Transformer blocks
        for _ in range(self.depth):
            x = Encoder1DBlock(mlp_dim=self.mlp_dim, num_heads=self.num_heads)(x)
            
        x = nn.LayerNorm()(x)
        
        # For simplicity, we pool the sequence to a single latent vector
        # A more advanced JEPA would predict sequence-to-sequence at the patch level.
        # But we pool it here to feed into the World Model.
        pooled = jnp.mean(x, axis=1)
        return pooled

class JEPAPredictor(nn.Module):
    """
    Lightweight predictor network (P) to map context representations to target representations.
    """
    latent_dim: int = 256
    depth: int = 2
    num_heads: int = 8
    mlp_dim: int = 1024
    
    @nn.compact
    def __call__(self, context_latents):
        # A small transformer or MLP predictor
        # For pooled representations, an MLP is often sufficient.
        x = context_latents
        for _ in range(self.depth):
            # Using simple Dense blocks for pooled predictor
            y = nn.Dense(self.mlp_dim)(x)
            y = nn.gelu(y)
            y = nn.Dense(self.latent_dim)(y)
            x = x + y
        return x
