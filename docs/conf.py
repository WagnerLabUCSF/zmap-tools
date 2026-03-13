# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# So autodoc can find the zmap package
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information ------------------------------------------------------
project = "zmap-tools"
copyright = "2025, Daniel Wagner, WagnerLab UCSF"
author = "Daniel Wagner"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ----------------------------------------------------

# Mock imports that aren't available at doc-build time
autodoc_mock_imports = [
    "anndata",
    "scanpy",
    "scipy",
    "sklearn",
    "tqdm",
    "adjustText",
    "symphonypy",
    "seaborn",
    "requests",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstrings = False
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# Intersphinx: link to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Suppress warnings for missing references to external types
nitpicky = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output --------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "navigation_depth": 3,
}

# Custom CSS
html_css_files = [
    "custom.css",
]

html_context = {
    "display_github": True,
    "github_user": "WagnerLabUCSF",
    "github_repo": "zmap-tools",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
