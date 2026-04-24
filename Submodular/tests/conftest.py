import os
import sys

# Allow `pytest tests/` from the Submodular directory without installing the package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
