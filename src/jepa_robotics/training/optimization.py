from ConfigSpace import (
    ConfigurationSpace,
    Integer,
    Float,
    Categorical,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
)
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration


def build_smac_scenario(
    run_name: str = "v1_normalised_probe",
) -> Scenario:
    """
    Constructs the SMAC3 Configuration Space for the V1 True-Masking + SIGReg architecture.

    Design Principles:
        - All ranges are calibrated against SOTA V-JEPA / ViT / MAE literature.
        - Forbidden clauses prevent architecturally invalid configurations
          (e.g. latent_dim indivisible by num_heads).
        - Hyperband budgets use clean powers-of-2 rungs: 4 → 8 → 16 epochs.
        - The search space is fresh: old smac3_output/ results are incompatible
          with this new architecture and should not be re-used.
    """
    cs = ConfigurationSpace(seed=42)

    # ── Perception Engine (V-JEPA ViT) ───────────────────────────────────────
    latent_dim = Categorical("latent_dim", [128, 256, 512], default=256)

    # Depth range follows canonical ViT-S (12) down to minimal (2).
    # For our 64-patch input (256px / 32px patch), depth 4–8 is the practical sweet spot.
    vit_depth = Integer("vit_depth", (2, 8), default=4)

    # patch_size=64 removed: with 256×256 images it yields only 16 patches total.
    # At masking_ratio=0.75 that leaves just 4 context patches — degenerate.
    patch_size = Categorical("patch_size", [16, 32], default=32)

    # True spatial masking ratio. V-JEPA paper uses 0.75–0.9.
    # Lower bound 0.5 allows SMAC to discover if easier problems benefit from more context.
    masking_ratio = Float("masking_ratio", (0.5, 0.90), default=0.75)

    # ── Shared Transformer Heads (ViT + Predictor + World Model) ─────────────
    # num_heads must evenly divide latent_dim. Forbidden clauses enforce this below.
    # head_dim = latent_dim / num_heads:
    #   128 / 4  = 32  ✓ | 256 / 4  = 64  ✓ | 512 / 4  = 128 ✓
    #   128 / 8  = 16  ✓ | 256 / 8  = 32  ✓ | 512 / 8  = 64  ✓
    #   128 / 16 = 8   ✗ (too small, attention degrades) | 256/16=16 ✓ | 512/16=32 ✓
    num_heads = Categorical("num_heads", [4, 8, 16], default=8)

    # ── World Model (Action-Conditioned Transformer) ──────────────────────────
    # Temporal reasoning depth. 4–6 layers is standard for causal sequence models
    # at this data scale.
    wm_depth = Integer("wm_depth", (2, 6), default=4)

    # ── Activation Function ───────────────────────────────────────────────────
    # gelu / silu are both SOTA for Transformers. relu included as a lower baseline.
    activation_fn = Categorical("activation_fn", ["gelu", "silu", "relu"], default="gelu")

    # ── EMA Target Encoder ───────────────────────────────────────────────────
    # JEPA papers typically use 0.996–0.9999. Lower tau = faster teacher updates
    # (more aggressive, riskier). Higher = more stable but slower signal propagation.
    tau = Float("tau", (0.990, 0.9999), default=0.996)

    # ── Optimiser & LR ───────────────────────────────────────────────────────
    # Core LR: V-JEPA / MAE / ViT literature peaks at 1.5e-4 to 3e-4 for ViT-S/B.
    # Log-scale search covers this well.
    learning_rate = Float("learning_rate", (5e-5, 5e-4), default=1.5e-4, log=True)

    # Probe LR: independent linear head, higher LR is appropriate.
    probe_learning_rate = Float("probe_learning_rate", (1e-4, 1e-2), default=1e-3, log=True)

    # Weight decay: SOTA ViT / JEPA training uses 0.04–0.1 (AdamW).
    # Old range (1e-6, 1e-2) capped below the effective SOTA region.
    weight_decay = Float("weight_decay", (1e-3, 0.1), default=0.04, log=True)

    # Batch size: RTX 5090 (32GB VRAM). True masking reduces peak memory vs full attention
    # (E_x sees only N_context << N_total tokens). 64 is now feasible and preferable.
    # Removed 8: too small for stable gradient estimates with cross-embodiment batching.
    batch_size = Categorical("batch_size", [16, 32, 64], default=32)

    # Temporal sequence length for the World Model.
    seq_len = Integer("seq_len", (3, 8), default=5)

    # ── Loss Balancing ────────────────────────────────────────────────────────
    # loss_alpha: weight on the temporal dynamics loss relative to the JEPA latent loss.
    # The patch-level JEPA loss is now larger in magnitude than the old pooled version,
    # so the effective balance has shifted. Keeping (0.1, 10.0) lets SMAC rediscover it.
    loss_alpha = Float("loss_alpha", (0.1, 10.0), default=1.0)

    # SIGReg weight: InfoMax entropy regularizer. Operates on pooled latents (B*S, D).
    # Effective range empirically 0.05–0.5; log-scale search from 0.01 to 1.0.
    sigreg_weight = Float("sigreg_weight", (0.01, 1.0), default=0.1, log=True)

    # ── Register All Hyperparameters ──────────────────────────────────────────
    cs.add([
        latent_dim, vit_depth, patch_size, masking_ratio,
        num_heads, wm_depth, activation_fn,
        tau, learning_rate, probe_learning_rate, weight_decay,
        batch_size, seq_len, loss_alpha, sigreg_weight,
    ])

    # ── Forbidden Configurations ──────────────────────────────────────────────
    # Prevent num_heads=16 when latent_dim=128: head_dim = 128/16 = 8, which is
    # below the minimum stable attention head dimension (~16). Attention degrades
    # to near-noise at head_dim < 16.
    cs.add(ForbiddenAndConjunction(
        ForbiddenEqualsClause(latent_dim, 128),
        ForbiddenEqualsClause(num_heads, 16),
    ))

    # ── SMAC3 Scenario ────────────────────────────────────────────────────────
    # Hyperband with clean power-of-2 rungs: 4 → 8 → 16 epochs (eta=2).
    # min_budget=4 respects the JEPA "bump" heuristic (representations physically
    # expand in early epochs — pruning before epoch 4 discards valid architectures).
    # max_budget=16 gives survivors a meaningful evaluation horizon.
    # n_trials=75: expanded from 50 to cover the enlarged search space.
    scenario = Scenario(
        cs,
        deterministic=True,   # JAX PRNG keys make evaluation reproducible
        n_trials=75,
        name=run_name,
        output_directory="smac3_output",
        min_budget=4,
        max_budget=16,
    )

    return scenario


def get_evaluation_function(do_eval: bool):
    """Returns the SMAC evaluation function, accepting a Hyperband budget parameter."""

    def evaluation_function(
        config: Configuration,
        seed: int = 0,
        budget: float = 16.0,
    ) -> float:
        """
        Called by SMAC to evaluate a given architecture configuration.

        Args:
            config: Hyperparameter configuration to evaluate.
            seed: Random seed (unused — JAX PRNG keys handle determinism).
            budget: Number of epochs to train (set by Hyperband).

        Returns:
            Weighted probe score (lower is better). NaN-diverged runs return 999.0.
        """
        from jepa_robotics.training.loop import train_model

        config_dict = dict(config)
        config_dict["is_smac_run"] = True

        epochs_to_run = int(budget)
        print(f"\n[SMAC] Evaluating config for {epochs_to_run} epochs | "
              f"latent={config_dict.get('latent_dim')} "
              f"vit_d={config_dict.get('vit_depth')} "
              f"patch={config_dict.get('patch_size')} "
              f"mask={config_dict.get('masking_ratio', 0):.2f} "
              f"lr={config_dict.get('learning_rate', 0):.2e} "
              f"sigreg={config_dict.get('sigreg_weight', 0):.3f}")

        loss = train_model(config_dict, num_epochs=epochs_to_run, do_eval=do_eval)
        return float(loss)

    return evaluation_function


def run_smac_optimization(do_eval: bool = True):
    """
    Executes the SMAC3 Hyperband optimization sweep.

    Hyperband schedule (eta=2):
        Rung 1:  4 epochs  — all n_trials configurations evaluated
        Rung 2:  8 epochs  — top 50% survivors promoted
        Rung 3: 16 epochs  — top 25% survivors evaluated to full budget

    The incumbent (best configuration found) is returned and printed.
    """
    from smac.intensifier.hyperband import Hyperband

    scenario = build_smac_scenario()

    intensifier = Hyperband(
        scenario,
        incumbent_selection="highest_observed_budget",
        eta=2,  # Clean power-of-2 halving: 4 → 8 → 16
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        get_evaluation_function(do_eval),
        intensifier=intensifier,
    )

    incumbent = smac.optimize()
    print(f"\n[SMAC] Optimization Complete.")
    print(f"[SMAC] Optimal Configuration: {incumbent}")
    return incumbent
