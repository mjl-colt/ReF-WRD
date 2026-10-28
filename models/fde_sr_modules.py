"""Backward-compatible exports for the paper-aligned ReF-WRD modules.

The original prototype in this path performed Stage-II correction in RGB
space. ReF-WRD corrects the detached Stage-I latent instead, so all maintained
entry points now resolve to models.ref_wrd_modules.
"""

from .ref_wrd_modules import (
    AdaptiveWaveletResidualModulator,
    FDE_SRLoss,
    FrequencyDecoupledPromptLearner,
    GaussianFrequencyDecomposition,
    LatentResidualCorrector,
    ReFWRDLoss,
    StageIIResidualCorrector,
    StationaryHaarWaveletTransform,
    high_frequency_l1_loss,
    wavelet_consistency_loss,
)


# The legacy name now denotes the non-decimated transform used by the paper.
HaarWaveletTransform = StationaryHaarWaveletTransform


__all__ = [
    "AdaptiveWaveletResidualModulator",
    "FDE_SRLoss",
    "FrequencyDecoupledPromptLearner",
    "GaussianFrequencyDecomposition",
    "HaarWaveletTransform",
    "LatentResidualCorrector",
    "ReFWRDLoss",
    "StageIIResidualCorrector",
    "StationaryHaarWaveletTransform",
    "high_frequency_l1_loss",
    "wavelet_consistency_loss",
]
