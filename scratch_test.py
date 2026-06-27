import jax
import jax.numpy as jnp
import optax
import math
from jepa_robotics.models.v_jepa import ViTEncoder, JEPAPredictor, StateLinearProbe
from jepa_robotics.models.world_model import ActionConditionedTransformer
from jepa_robotics.training.step import create_steps

rng = jax.random.PRNGKey(0)
encoder_def = ViTEncoder(patch_size=16, latent_dim=256, depth=7, num_heads=16)
predictor_def = JEPAPredictor(latent_dim=256, depth=2, num_heads=16)
wm_def = ActionConditionedTransformer(latent_dim=256, depth=4, num_heads=8)
probe_def = StateLinearProbe(out_dim=10)

core_opt = optax.chain(optax.clip(1.0), optax.adamw(1e-4))
probe_opt = optax.chain(optax.clip(1.0), optax.adamw(1e-4))

train_step, eval_step = create_steps(
    encoder_def, predictor_def, wm_def, probe_def, core_opt, probe_opt,
    loss_alpha=1.0, patch_size=16, image_size=256, masking_ratio=0.75,
    sigreg_weight=0.5, use_amp=True
)

init_rng, step_rng = jax.random.split(rng)
mock_img = jnp.ones((2, 6, 256, 256, 3))
mock_action = jnp.ones((2, 6, 10))
mock_state = jnp.ones((2, 6, 10))

enc_params = encoder_def.init({'params': init_rng}, mock_img.reshape((2*6, 256, 256, 3)))
pred_params = predictor_def.init(init_rng, jnp.ones((2*6, 64, 256)), jnp.ones((2*6, 192, 256)))
wm_params = wm_def.init(init_rng, jnp.ones((2, 5, 256)), jnp.ones((2, 5, 10)))
probe_params = probe_def.init(init_rng, jnp.ones((2, 6, 256)))

state = {
    "encoder_params": enc_params,
    "predictor_params": pred_params,
    "wm_params": wm_params,
    "probe_params": probe_params,
    "target_params": enc_params,
    "core_opt_state": core_opt.init((enc_params, pred_params, wm_params)),
    "probe_opt_state": probe_opt.init(probe_params),
    "rng": step_rng
}

batch = {"image": mock_img, "action_10d": mock_action, "state_10d": mock_state}

print("Running 300 steps...")
for i in range(300):
    state, metrics = train_step(state, batch, tau=0.995)
    if i % 50 == 0 or i == 299:
        print(f"Step {i}: Avg L={metrics['loss']:.4f} Pos={metrics['pos_mse']:.4f} Rot={metrics['rot_mse']:.4f} sig_reg={metrics['sig_reg']:.4f}")
        if jnp.isnan(metrics['loss']):
            print("NaN detected!")
            break
