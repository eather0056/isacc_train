"""Compatibility patch for Isaac Sim's bundled torch."""

try:
    import torch.multiprocessing.reductions as torch_reductions
    from multiprocessing.reduction import ForkingPickler as MPForkingPickler

    # Isaac Sim's torch bundle omits ForkingPickler; torchrl expects it.
    if not hasattr(torch_reductions, "ForkingPickler"):
        torch_reductions.ForkingPickler = MPForkingPickler
except Exception:
    # If torch isn't importable yet, skip silently.
    pass
