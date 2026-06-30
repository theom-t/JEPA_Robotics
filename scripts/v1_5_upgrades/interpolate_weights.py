import os
import sys
import jax
import jax.numpy as jnp
import flax.serialization

def interpolate_pos_embedding(pos_emb, original_grid_size=(16, 16), target_grid_size=(32, 32)):
    # pos_emb shape is (1, 256, D)
    b, n_patches, d = pos_emb.shape
    assert n_patches == original_grid_size[0] * original_grid_size[1], "Patch count mismatch!"
    
    # Reshape to 2D spatial grid (1, H, W, D)
    grid = pos_emb.reshape((b, original_grid_size[0], original_grid_size[1], d))
    
    # Interpolate using jax.image.resize (bicubic)
    target_shape = (b, target_grid_size[0], target_grid_size[1], d)
    new_grid = jax.image.resize(grid, shape=target_shape, method='bicubic')
    
    # Flatten back to 1D patch sequence (1, 1024, D)
    new_pos_emb = new_grid.reshape((b, target_grid_size[0] * target_grid_size[1], d))
    return new_pos_emb

def main():
    input_path = "/home/tmainetucker/Repos/JEPA_Robotics/models/v1_jepa_backbone/v1_final_weights.msgpack"
    output_path = "/home/tmainetucker/Repos/JEPA_Robotics/models/v1_jepa_backbone/v1_5_initial_weights.msgpack"
    
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}")
        sys.exit(1)
        
    print(f"Loading V1 weights from: {input_path}")
    with open(input_path, "rb") as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
        
    # Interpolate Context Encoder
    if 'encoder_params' in state_dict and 'pos_embedding' in state_dict['encoder_params']['params']:
        old_emb = state_dict['encoder_params']['params']['pos_embedding']
        print(f"Interpolating Context Encoder pos_embedding from {old_emb.shape}...")
        new_emb = interpolate_pos_embedding(old_emb)
        state_dict['encoder_params']['params']['pos_embedding'] = new_emb
        print(f"  -> New shape: {new_emb.shape}")
        
    # Interpolate Target Encoder
    if 'target_params' in state_dict and 'pos_embedding' in state_dict['target_params']['params']:
        old_emb = state_dict['target_params']['params']['pos_embedding']
        print(f"Interpolating Target Encoder pos_embedding from {old_emb.shape}...")
        new_emb = interpolate_pos_embedding(old_emb)
        state_dict['target_params']['params']['pos_embedding'] = new_emb
        print(f"  -> New shape: {new_emb.shape}")
        
    with open(output_path, "wb") as f:
        f.write(flax.serialization.msgpack_serialize(state_dict))
        
    print(f"\\n[SUCCESS] Successfully interpolated V1 weights to V1.5 High-Acuity format.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
