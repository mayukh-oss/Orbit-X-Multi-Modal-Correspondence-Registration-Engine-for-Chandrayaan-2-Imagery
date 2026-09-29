# [ANNOTATION] Main entry point module executed when running `python -m sih26166.app` or `python -m sih26166`.
# [ANNOTATION] Docstring providing entry point information.
"""
Direct entry point for running `python -m sih26166.app` or `python -m sih26166`.
"""

# [ANNOTATION] Import the sys module to interact with system-level functions like exit codes and arguments.
import sys

# [ANNOTATION] Print startup message indicating execution via __main__.
print("[STARTUP] Invoking sih26166 entry point from __main__.py...")

# [ANNOTATION] Import the main application launcher function from the GUI package module.
from sih26166.app.gui import main

# [ANNOTATION] Check if this script is being executed directly as the main module.
if __name__ == "__main__":
    # [ANNOTATION] Print log confirming execution threshold passed.
    print("[STARTUP] Main block triggered. Executing main() application loop...")
    # [ANNOTATION] Execute the main GUI function and exit the process using its return code.
    sys.exit(main())