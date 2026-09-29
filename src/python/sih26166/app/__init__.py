# [ANNOTATION] Package initialization file for the SIH26166 GUI Application Package.
# [ANNOTATION] This file defines what is exposed when importing `sih26166.app`.

# [ANNOTATION] Print a diagnostic log entry indicating package initialization.
print("[INIT] Initializing sih26166.app package...")

# [ANNOTATION] Docstring describing the purpose of the application package.
"""
SIH26166 GUI Application Package.
"""

# [ANNOTATION] Import the MainWindow class and main entry function from the local gui module.
from .gui import MainWindow, main

# [ANNOTATION] Define __all__ to explicitly declare public symbols exported by this package.
__all__ = ["MainWindow", "main"]

# [ANNOTATION] Print confirmation that package exports are configured.
print(f"[INIT] Package sih26166.app initialized. Exported symbols: {__all__}")