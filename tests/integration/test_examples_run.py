"""§6 safety net: every example's ``main()`` runs to completion with its default options.

A deliberately shallow guard for the frozen public API (architecture plan §4). It asserts only
that each example *runs* end to end under its default settings -- it inspects no return value, so
it stays valid as the work packages legitimately change the numbers. The defaults are light enough
to run in a few seconds each (the heaviest, ``r_adptativity_poc``, is ~13 s). The two 2-D examples
mesh with gmsh, which needs ``libGLU``; where that shared library is absent the example is skipped
rather than failed.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

os.environ.setdefault(
    "MPLBACKEND", "Agg"
)  # plt.show() becomes a no-op -> no blocking windows

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"

# Every example. The ones that mesh with gmsh are flagged so they can be skipped when gmsh / libGLU
# cannot be loaded (e.g. headless CI without the GL stack).
ALL = [
    "1d_beam_hermite",
    "1d_bar_deflection_pgd",
    "design_poc",
    "r_adptativity_poc",
    "benchmark",
    "2d",
    "2d_r_adaptivity",
]
GMSH_EXAMPLES = {"2d", "2d_r_adaptivity"}


def _load(name):
    """Import an example's ``main.py`` as a module (without executing ``main()``)."""
    path = EXAMPLES / name / "main.py"
    spec = importlib.util.spec_from_file_location(f"ex_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gmsh_available():
    """True if gmsh (and its ``libGLU`` dependency) can be imported in this environment."""
    try:
        import gmsh  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.parametrize("name", ALL)
def test_example_runs(name, tmp_path, monkeypatch):
    """Each example's ``main()`` completes with default options (no return value is inspected).

    This is the broadest, shallowest guard in the suite: it drives the full public-API surface
    every example touches -- meshes, function spaces, fields, assembly, training -- end to end, and
    fails if a refactor breaks that surface. It asserts nothing about the *result*, so it needs no
    update when a work package changes the numbers; the numerical guards live in the dedicated
    unit/integration tests instead. Two fixtures keep the run hermetic: ``sys.argv`` is cleared so
    argparse-based examples (``benchmark``) fall back to their defaults rather than parsing pytest's
    command line, and the working directory is a scratch path so figures some examples save do not
    litter the tree. The gmsh-backed 2-D examples are skipped where ``libGLU`` is unavailable.
    """
    if name in GMSH_EXAMPLES and not _gmsh_available():
        pytest.skip("gmsh/libGLU unavailable in this environment")
    monkeypatch.setattr(sys, "argv", [name])  # argparse examples -> their defaults
    monkeypatch.chdir(tmp_path)  # saved figures land in a scratch dir
    _load(name).main()
