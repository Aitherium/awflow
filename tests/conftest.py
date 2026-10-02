"""Import the package FROM THE TREE, the awrouter/awmine tests idiom.

Without this the suite needs `pip install -e` before it can even be collected, and
the hermetic CI gate installs nothing -- a runner whose python happened to carry an
editable install passed, a fresh one errored at collection (measured 2026-10-02).
"""
import sys as _sys
from pathlib import Path as _Path

_PKG_ROOT = _Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PKG_ROOT))
