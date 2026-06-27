import flax.serialization
import sys
import os
import pytest

def test_load():
    msgpack_path = "/home/tmainetucker/Repos/JEPA_Robotics/checkpoints/v1_jepa_backbone/v1_weights.msgpack"
    
    if not os.path.exists(msgpack_path):
        pytest.skip(f"File not found at {msgpack_path}. Run training first.")

    print(f"Loading weights from {msgpack_path}...")
    
    try:
        with open(msgpack_path, "rb") as f:
            loaded_bytes = f.read()
            
        loaded_state = flax.serialization.msgpack_restore(loaded_bytes)
        
        print("\n✅ Successfully restored msgpack dictionary!")
        print(f"Found {len(loaded_state.keys())} root keys in checkpoint:")
        
        for k in loaded_state.keys():
            print(f"  - {k}")
            
        print("\nInspecting structure inside 'encoder_params':")
        # Flax puts everything under a 'params' sub-dictionary
        if 'params' in loaded_state['encoder_params']:
            encoder_layers = list(loaded_state['encoder_params']['params'].keys())
            print(f"  Encoder has {len(encoder_layers)} parameter groups (layers).")
            
            # Show the first layer parameter shapes as proof of life
            first_layer = encoder_layers[0]
            for param_name, param_tensor in loaded_state['encoder_params']['params'][first_layer].items():
                # Tensors restored via msgpack without an initialization template are pure dicts/lists of primitive types.
                # However, Flax restore logic handles it. Let's see what type it is.
                print(f"    -> {first_layer}/{param_name} loaded as type: {type(param_tensor)}")
        else:
            print("  Warning: No 'params' sub-key found in encoder. Structure might be different than expected.")

    except Exception as e:
        pytest.fail(f"Failed to load weights: {e}")

if __name__ == "__main__":
    test_load()
