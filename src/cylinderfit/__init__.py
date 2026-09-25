"""Deprecated compatibility alias for :mod:`cylfit`.

``import cylinderfit`` still works but will be removed in a future release;
use ``import cylfit`` instead.
"""

import warnings

from cylfit import *  # noqa: F401,F403
from cylfit import __all__, __version__  # noqa: F401

warnings.warn(
    "The 'cylinderfit' package has been renamed to 'cylfit'; "
    "'import cylinderfit' is deprecated and will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
