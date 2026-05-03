"""Sphinx configuration for cylfit documentation."""

import sys
from pathlib import Path

# Make the package importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project info -----------------------------------------------------------
project = "cylfit"
copyright = "2026, cylfit contributors"
author = "cylfit contributors"
release = "0.1.0"

# -- Extensions -------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # NumPy/Google docstring support
    "sphinx.ext.viewcode",        # [source] links
    "sphinx.ext.intersphinx",     # Cross-links to NumPy, Python docs
    "sphinx.ext.mathjax",         # Math rendering
]

# Try optional extensions, skip gracefully if not installed
try:
    import sphinx_autodoc_typehints  # noqa: F401
    extensions.append("sphinx_autodoc_typehints")
except ImportError:
    pass

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output ------------------------------------------------------------
html_theme = "furo"
html_title = "cylfit"
html_theme_options = {
    "navigation_with_keys": True,
}

# -- General ----------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = []
