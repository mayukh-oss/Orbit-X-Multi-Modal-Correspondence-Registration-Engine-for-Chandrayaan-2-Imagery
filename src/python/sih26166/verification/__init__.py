# [ANNOTATION] Package initialization file for geometric verification utilities.
"""
Geometric verification for image correspondences.

This package separates descriptor-level matching from geometric consistency
estimation. The latter is responsible for identifying spatially consistent
inliers and rejecting geometrically incompatible matches.
"""

# [ANNOTATION] Print diagnostic log entry indicating package initialization.
print("[GEOMETRIC_INIT] Initializing sih26166.verification.geometric package...")

# [ANNOTATION] Import core geometric verification symbols from local geometric module.
from .geometric import (
    GeometricModel,
    GeometricVerificationConfig,
    GeometricVerificationResult,
    verify_geometry,
)

# [ANNOTATION] Define public package exports.
__all__ = [
    "GeometricModel",
    "GeometricVerificationConfig",
    "GeometricVerificationResult",
    "verify_geometry",
]

# [ANNOTATION] Log exported symbols.
print(f"[GEOMETRIC_INIT] Package loaded. Exported symbols: {__all__}")