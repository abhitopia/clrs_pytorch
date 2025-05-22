from .base import ProcessorBase, GraphFeatures
from .gat import GAT, GATFull, GATv2, GATv2Full
from .pgn import PGN, PGNMask, DeepSets, MPNN, Reduction
from enum import Enum
from typing import Optional

class Processor(str, Enum):
    gat = "gat"
    gat_full = "gat_full"
    gat_v2 = "gat_v2"
    gat_v2_full = "gat_v2_full"
    pgn = "pgn"
    pgn_mask = "pgn_mask"
    deepsets = "deepsets"
    mpnn = "mpnn"
    triplet_mpnn = "triplet_mpnn"
    triplet_pgn = "triplet_pgn"
    triplet_pgn_mask = "triplet_pgn_mask"
    gpgn = "gpgn"
    gpgn_mask = "gpgn_mask"
    gmpnn = "gmpnn"
    triplet_gpgn = "triplet_gpgn"
    triplet_gpgn_mask = "triplet_gpgn_mask"
    triplet_gmpnn = "triplet_gmpnn"

    def __call__(self, 
                 hidden_dim: int = 128, 
                 use_ln: bool = True, 
                 nb_heads: Optional[int] = 1, 
                 triplet_fts_size: Optional[int] = 8,
                 reduction: Reduction = Reduction.MAX,
                 mp_steps: int = 1) -> ProcessorBase:
        common_kwargs = {
            "node_feat_size": hidden_dim,
            "hidden_size": hidden_dim,
            "use_layer_norm": use_ln,
            "mp_steps": mp_steps,
        }
        pgn_common_kwargs = {
            "msgs_mlp_sizes": [hidden_dim, hidden_dim],
            "triplet_feature_size": triplet_fts_size,
        }
        if self == Processor.gat:
            return GAT(**common_kwargs,
                       nb_heads=nb_heads)
        elif self == Processor.gat_full:
            return GATFull(**common_kwargs,
                            nb_heads=nb_heads)
        elif self == Processor.gat_v2:
            return GATv2(**common_kwargs,
                         nb_heads=nb_heads)
        elif self == Processor.gat_v2_full:
            return GATv2Full(**common_kwargs,
                             nb_heads=nb_heads)
        elif self == Processor.deepsets:
            return DeepSets(**common_kwargs,
                            **pgn_common_kwargs,
                            reduction=reduction,
                            use_triplets=False,
                            gated=False)
        elif self == Processor.mpnn:
            return MPNN(**common_kwargs,
                        **pgn_common_kwargs,
                        reduction=reduction,
                        use_triplets=False,
                        gated=False)
        elif self == Processor.pgn:
            return PGN(**common_kwargs,
                       **pgn_common_kwargs,
                       reduction=reduction,
                       use_triplets=False,
                       gated=False)
        elif self == Processor.pgn_mask:
            return PGNMask(**common_kwargs,
                            **pgn_common_kwargs,
                            reduction=reduction,
                            use_triplets=False,
                            gated=False)
        elif self == Processor.triplet_mpnn:
            return MPNN(**common_kwargs,
                        **pgn_common_kwargs,
                        reduction=reduction,
                        use_triplets=True,
                        gated=False)
        elif self == Processor.triplet_pgn:
            return PGN(**common_kwargs,
                       **pgn_common_kwargs,
                       reduction=reduction,
                       use_triplets=True,
                       gated=False)
        elif self == Processor.triplet_pgn_mask:
            return PGNMask(**common_kwargs,
                            **pgn_common_kwargs,
                            reduction=reduction,
                            use_triplets=True,
                            gated=False)
        elif self == Processor.gpgn:
            return PGN(**common_kwargs,
                       **pgn_common_kwargs,
                       reduction=reduction,
                       use_triplets=False,
                       gated=True)
        elif self == Processor.gpgn_mask:
            return PGNMask(**common_kwargs,
                            **pgn_common_kwargs,
                            reduction=reduction,
                            use_triplets=False,
                            gated=True)
        elif self == Processor.gmpnn:
            return MPNN(**common_kwargs,
                        **pgn_common_kwargs,
                        reduction=reduction,
                        use_triplets=False,
                        gated=True)
        elif self == Processor.triplet_gpgn:
            return PGN(**common_kwargs,
                       **pgn_common_kwargs,
                       reduction=reduction,
                       use_triplets=True,
                       gated=True)
        elif self == Processor.triplet_gpgn_mask:
            return PGNMask(**common_kwargs,
                           **pgn_common_kwargs,
                           reduction=reduction,
                           use_triplets=True,
                           gated=True)
        elif self == Processor.triplet_gmpnn:
            return MPNN(**common_kwargs,
                        **pgn_common_kwargs,
                        reduction=reduction,
                        use_triplets=True,
                        gated=True)
        else:
            raise ValueError(f"Processor {self} not implemented")
__all__ = ["ProcessorBase", "Processor", "GraphFeatures"]