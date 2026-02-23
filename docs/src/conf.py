# conf.py — Sphinx configuration for python-template

import os
import sys

# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------
project = "neurom"
author = "Alexandre Daby, et al. [TODO]"
copyright = "2026, Alexandre Daby"
release = "0.1.0"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx_autodoc_typehints",  # Auto-links type hints
    "myst_parser",  # Markdown support
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Fancy UI elements
    "autoapi.extension",  # AutoAPI for automatic API docs
]

# MyST (Markdown) configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

# Napoleon (Google docstrings) configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True,
    "exclude-members": "__init__,__dataclass_fields__",
}
autodoc_typehints = "signature"
python_use_unqualified_type_names = True

# AutoAPI configuration — scan the installed package
autoapi_type = "python"
autoapi_dirs = ["../../src/neurom"]  # Empty because we use the installed package
autoapi_root = "api"  # AutoAPI output directory
autoapi_add_toctree_entry = True  # Include in TOC automatically
autoapi_keep_files = False  # Keep generated Markdown files for inspection
autoapi_generate_api_docs = True
autoapi_member_order = "bysource"  # or 'alphabetical' or 'bysource'

# Add this template option to control what gets individual pages
autoapi_template_dir = None  # Use default templates
autoapi_own_page_level = "module"  # or "package"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
]

# HTML output
html_theme = "sphinx_rtd_theme"
html_title = project
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Intersphinx
extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}


# Skip certain members in module pages
def autodoc_skip_member(app, what, name, obj, skip, options):
    # Skip class members when documenting the module
    if what == "module":
        try:
            if hasattr(obj, "__qualname__") and "." in obj.__qualname__:
                return True
        except Exception:
            pass
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
