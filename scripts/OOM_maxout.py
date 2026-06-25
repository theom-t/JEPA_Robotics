"""
OOM Max-Out Stress Test for the True-Masking V-JEPA + SIGReg Architecture.

Purpose:
    Verify the RTX 5090 (32GB VRAM) can handle the absolute worst-case configuration
    within the SMAC3 search space BEFORE launching the full optimization sweep.
    Catching an OOM here is cheap; catching it 20 trials into a SMAC sweep is very expensive.

Worst-Case Memory Analysis (True Spatial Masking):
    The peak memory bottleneck is driven by three concurrent allocations:
      1. E_y (Target Encoder): processes ALL N_patches tokens per image.
         patch_size=16 → 256 patches × batch_size×seq_len images simultaneously.
      2. JEPAPredictor: Transformer over (N_context + N_target) = N_patches tokens.
         With masking_ratio=0.5, context=128, target=128 → 256 tokens @ latent_dim=512.
      3. World Model: causal attention over seq_len=8 pooled latents.

    Combined with latent_dim=512, num_heads=16, batch_size=64, the allocations are
    substantial but should remain within the 32GB budget.
"""

import os
import sys
import subprocess

# Suppress XLA C++ warnings and prevent JAX from searching for TPUs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Force XLA to pre-allocate — ensures we get a true OOM crash if we exceed physical VRAM.
# Without this, JAX uses lazy allocation and may silently spill to system RAM.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from jepa_robotics.training.loop import train_model


def get_vram_usage() -> str:
    """Query nvidia-smi for current GPU VRAM usage."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        reports = []
        for i, line in enumerate(lines):
            used, total, free = [int(x.strip()) for x in line.split(",")]
            pct = (used / total) * 100
            reports.append(
                f"  GPU {i}: {used:,} MiB used / {total:,} MiB total ({pct:.1f}%) — {free:,} MiB free"
            )
        return "\n".join(reports)
    except Exception as e:
        return f"  [nvidia-smi unavailable: {e}]"


def run_config_test(label: str, config: dict, epochs: int = 1) -> bool:
    """
    Run a single configuration test and return True on success, False on OOM.

    Args:
        label: Human-readable description of this configuration.
        config: Hyperparameter dict to pass to train_model.
        epochs: Number of epochs to run (1 is sufficient to trigger full JIT + alloc).

    Returns:
        True if the run succeeded without OOM.
    """
    print(f"\n{'─' * 80}")
    print(f"  TEST: {label}")
    print(f"{'─' * 80}")
    for key, value in sorted(config.items()):
        if key not in ("disable_wandb", "is_smac_run"):
            print(f"    {key:<25} = {value}")
    print()
    print("  VRAM before JIT compilation:")
    print(get_vram_usage())
    print()

    try:
        train_model(config, num_epochs=epochs, do_eval=False)
        print("\n  VRAM after forward pass:")
        print(get_vram_usage())
        print(f"\n  ✅ PASS — '{label}' completed without OOM.\n")
        return True
    except (MemoryError, RuntimeError, Exception) as e:
        if "out of memory" in str(e).lower() or "resource exhausted" in str(e).lower():
            print(f"\n  ❌ OOM — '{label}' exceeded VRAM budget.")
            print(f"     Error: {e}\n")
        else:
            print(f"\n  ❌ UNEXPECTED ERROR in '{label}':")
            print(f"     {type(e).__name__}: {e}\n")
            raise
        return False


def run_oom_test() -> None:
    """Execute the full OOM stress test suite."""
    print("=" * 80)
    print("  🚀  V-JEPA TRUE-MASKING + SIGReg  —  OOM MAX-OUT STRESS TEST  🚀")
    print("=" * 80)
    print("  Pre-sweep VRAM validation for the SMAC3 search space bounds.")
    print(f"  Initial GPU state:")
    print(get_vram_usage())
    print("=" * 80)

    # ── Shared boilerplate fields ─────────────────────────────────────────────
    base = {
        "activation_fn": "gelu",
        "tau": 0.996,
        "learning_rate": 1.5e-4,
        "probe_learning_rate": 1e-3,
        "weight_decay": 0.04,
        "loss_alpha": 1.0,
        "sigreg_weight": 0.1,
        "disable_wandb": True,
        "is_smac_run": True,   # Uses a small data slice — tests OOM, not training quality
    }

    results = {}

    # ── TEST 1: Absolute Maximum Configuration ────────────────────────────────
    # This is the worst-case memory scenario across ALL SMAC search space bounds:
    #   patch_size=16  → 256 patches total (most tokens through E_y's full-image path)
    #   masking_ratio=0.5 → 128 context patches (largest E_x + Predictor sequence)
    #   latent_dim=512, num_heads=16 → largest embedding, fine-grained attention
    #   batch_size=64, seq_len=8 → maximum batch and temporal context
    results["max_config"] = run_config_test(
        label="Absolute Maximum (all SMAC bounds at worst-case memory)",
        config={
            **base,
            "latent_dim": 512,
            "vit_depth": 8,
            "patch_size": 16,          # 256 patches — memory bottleneck driver
            "masking_ratio": 0.5,      # 50% masked → 128 context tokens in E_x
            "wm_depth": 6,
            "num_heads": 16,
            "batch_size": 64,
            "seq_len": 8,
        },
    )

    # ── TEST 2: Realistic Heavy Config (likely SMAC favourite territory) ──────
    # patch_size=32 reduces token count 4× vs patch_size=16. This is where
    # the optimal config most likely lives given the edge-deployment constraints.
    results["realistic_heavy"] = run_config_test(
        label="Realistic Heavy (patch=32, latent=512, batch=64)",
        config={
            **base,
            "latent_dim": 512,
            "vit_depth": 8,
            "patch_size": 32,          # 64 patches — 4× fewer than patch_size=16
            "masking_ratio": 0.75,     # Standard V-JEPA masking: 16 context patches
            "wm_depth": 6,
            "num_heads": 16,
            "batch_size": 64,
            "seq_len": 8,
        },
    )

    # ── TEST 3: Edge Case — High Masking at patch_size=16 ────────────────────
    # masking_ratio=0.9 + patch_size=16 → only 25 context patches in E_x.
    # Predictor still processes 256 tokens (25 context + 230 target mask tokens).
    # This tests whether the Predictor Transformer's full-sequence attention OOMs.
    results["high_masking_fine_patch"] = run_config_test(
        label="High masking + fine patch (patch=16, mask=0.9, latent=256)",
        config={
            **base,
            "latent_dim": 256,
            "vit_depth": 6,
            "patch_size": 16,
            "masking_ratio": 0.90,     # 90% masked → 25 context, 230 target tokens
            "wm_depth": 4,
            "num_heads": 8,
            "batch_size": 32,
            "seq_len": 8,
        },
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("=" * 80)
    print("  📊  STRESS TEST RESULTS SUMMARY")
    print("=" * 80)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL (OOM)"
        print(f"  {name:<35}  {status}")
        if not passed:
            all_passed = False

    print()
    print("  Final GPU state:")
    print(get_vram_usage())
    print("=" * 80)

    if all_passed:
        print("  ✅ ALL TESTS PASSED — Safe to launch SMAC3 optimization sweep.")
        print("     Run:  python scripts/train.py --mode optimize")
    else:
        print("  ❌ ONE OR MORE TESTS FAILED.")
        print("     Constrain the failing hyperparameter bounds in optimization.py before sweeping.")
    print("=" * 80)


if __name__ == "__main__":
    run_oom_test()
