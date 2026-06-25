import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple


class MLPBlock(nn.Module):
    """A standard MLP block used in Vision Transformers."""

    mlp_dim: int
    out_dim: int
    activation_fn: str = "gelu"

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.mlp_dim)(inputs)
        act = getattr(nn, self.activation_fn)
        x = act(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class Encoder1DBlock(nn.Module):
    """Standard Transformer encoder block with pre-LayerNorm and residual connections."""

    mlp_dim: int
    num_heads: int
    activation_fn: str = "gelu"

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Self-attention block
        x = nn.LayerNorm()(inputs)
        x = nn.SelfAttention(num_heads=self.num_heads)(x)
        x = x + inputs

        # MLP block
        y = nn.LayerNorm()(x)
        y = MLPBlock(
            mlp_dim=self.mlp_dim,
            out_dim=inputs.shape[-1],
            activation_fn=self.activation_fn,
        )(y)
        return x + y


class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) Encoder for V-JEPA with true spatial patch masking.

    Supports two operating modes:
      - **Target Encoder (E_y)**: ``patch_indices=None`` — processes all N patches.
        Updated via EMA, not backprop. Provides the ground-truth latent targets.
      - **Context Encoder (E_x)**: ``patch_indices=<context_indices>`` — processes
        only the visible (unmasked) subset of patches. The encoder is blind to the
        masked region, forcing the Predictor to reason about missing spatial content.

    Critically, positional embeddings are added to ALL patch tokens BEFORE the
    subset selection. This ensures each context patch carries correct spatial
    position information into the Transformer, even though target patches are absent.
    """

    latent_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_dim: int = 1024
    patch_size: int = 16
    activation_fn: str = "gelu"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        patch_indices: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: Image tensor of shape ``(B, H, W, C)``.
            patch_indices: Optional 1-D integer array ``(N_keep,)`` of patch indices
                to process. If ``None``, all patches are processed (target path).
                If provided, only those patches are encoded (context path).

        Returns:
            patch_latents: ``(B, N_keep_or_N_all, D)`` per-patch representations.
            pooled: ``(B, D)`` mean-pooled representation for the World Model / Probe.
        """
        b, h, w, c = x.shape

        # 1. Patch embedding via strided convolution
        x = nn.Conv(
            features=self.latent_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )(x)
        x = x.reshape((b, -1, self.latent_dim))  # (B, N_patches, D)

        num_patches = x.shape[1]

        # 2. Add positional embeddings to ALL patches BEFORE masking.
        #    This preserves correct spatial position info in every kept patch token.
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches, self.latent_dim),
        )
        x = x + pos_embedding  # (B, N_patches, D)

        # 3. True Spatial Masking: select ONLY the context patches.
        #    This leaves the encoder completely blind to the target region.
        if patch_indices is not None:
            x = x[:, patch_indices, :]  # (B, N_keep, D)

        # 4. Transformer blocks over the (possibly masked) patch sequence
        for _ in range(self.depth):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                activation_fn=self.activation_fn,
            )(x)

        x = nn.LayerNorm()(x)

        # 5. Pool across the patch sequence for World Model / Probe consumption
        pooled = jnp.mean(x, axis=1)  # (B, D)

        return x, pooled


class JEPAPredictor(nn.Module):
    """
    Lightweight Predictor Network (P) for True V-JEPA.

    Given the encoded context patches from E_x and the positional embeddings for
    the masked target locations, it predicts what E_y (the EMA target encoder)
    would have produced at those masked positions.

    Architecture:
        1. Learnable mask tokens are placed at each target position.
        2. The target positional embeddings (extracted from the encoder's param table)
           are added to the mask tokens so the predictor knows *where* to predict.
        3. Context latents and positioned mask tokens are concatenated into a joint
           sequence and processed by a small Transformer.
        4. Only the output tokens corresponding to target positions are returned.

    This design forces the Predictor to learn to spatially interpolate/extrapolate
    physical scene structure from the visible context region.
    """

    latent_dim: int = 256
    depth: int = 2
    num_heads: int = 4
    mlp_dim: int = 512
    activation_fn: str = "gelu"

    @nn.compact
    def __call__(
        self,
        context_latents: jnp.ndarray,
        target_pos_embeddings: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            context_latents: ``(B, N_context, D)`` encoded context patch latents from E_x.
            target_pos_embeddings: ``(B, N_target, D)`` positional embeddings for the
                masked target patch positions (sliced from the encoder's pos_embedding param).

        Returns:
            predicted_target_latents: ``(B, N_target, D)`` predictions to compare against
                the true target encoder latents at those positions.
        """
        b, n_context, _ = context_latents.shape
        n_target = target_pos_embeddings.shape[1]

        # Learnable mask token — a single shared vector broadcast to all target positions
        mask_token = self.param(
            "mask_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.latent_dim),
        )

        # Inject spatial position into each mask token
        target_tokens = (
            jnp.broadcast_to(mask_token, (b, n_target, self.latent_dim))
            + target_pos_embeddings
        )  # (B, N_target, D)

        # Concatenate: [context patches | masked target tokens]
        # The Transformer attends across both to reason: "given visible context,
        # predict what the target encoder saw at these masked positions."
        x = jnp.concatenate([context_latents, target_tokens], axis=1)  # (B, N_context + N_target, D)

        # Small predictor Transformer
        for _ in range(self.depth):
            x = Encoder1DBlock(
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                activation_fn=self.activation_fn,
            )(x)

        x = nn.LayerNorm()(x)

        # Return ONLY the target position outputs — these are the predictions
        return x[:, n_context:, :]  # (B, N_target, D)


class StateLinearProbe(nn.Module):
    """
    Auxiliary linear probe to regress the 10D physical robot state (XYZ + 6D Rotation + Gripper)
    from the abstract pooled latent representations. Operates on a stop-gradient copy of the
    context encoder's pooled output, providing an interpretable diagnostic metric without
    polluting the self-supervised JEPA learning signal.
    """

    out_dim: int = 10

    @nn.compact
    def __call__(self, latents: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            latents: ``(B, D)`` or ``(B, S, D)`` pooled latent representations.

        Returns:
            Predicted robot state of shape ``(B, out_dim)`` or ``(B, S, out_dim)``.
        """
        return nn.Dense(self.out_dim)(latents)
