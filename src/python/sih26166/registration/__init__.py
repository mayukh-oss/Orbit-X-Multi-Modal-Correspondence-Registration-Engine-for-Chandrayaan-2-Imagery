# [ANNOTATION] Package initialization file for registration package exposure.
"""
Image registration for SIH26166.

This package provides transformation estimation and image warping utilities
for registering Chandrayaan-2 optical imagery and controlled validation data.

SIH 2026 | PS ID: SIH26166
"""

# [ANNOTATION] Print diagnostic log entry indicating package loading.
print("[REGISTRATION_INIT] Initializing sih26166.registration package...")

# [ANNOTATION] Import core registration classes and functions from local register module.
from sih26166.registration.register import (
    RegistrationConfig,
    RegistrationResult,
    register_image,
)

# [ANNOTATION] Define explicit public symbols exported by this package.
__all__ = [
    "RegistrationConfig",
    "RegistrationResult",
    "register_image",
]

# [ANNOTATION] Log registration package exports.
print(f"[REGISTRATION_INIT] Package sih26166.registration loaded. Exported symbols: {__all__}")