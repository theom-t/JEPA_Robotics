import jax

def update_target_ema(context_params, target_params, tau: float):
    """
    Updates the target encoder parameters using an Exponential Moving Average
    of the context encoder parameters.
    
    Equation: target_params = tau * target_params + (1 - tau) * context_params
    
    Args:
        context_params: PyTree of current context encoder parameters (E_x).
        target_params: PyTree of current target encoder parameters (E_y).
        tau: EMA decay rate (e.g., 0.996). Closer to 1.0 = slower update.
        
    Returns:
        New PyTree of updated target parameters.
    """
    # jax.tree_util.tree_map applies the lambda function to every corresponding 
    # leaf tensor in the two PyTrees simultaneously.
    return jax.tree_util.tree_map(
        lambda c, t: tau * t + (1.0 - tau) * c,
        context_params,
        target_params
    )
