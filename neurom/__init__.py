from .main import main
from importlib.metadata import version, PackageNotFoundError
__version__ = version("neurom")

__all__ = [main,__version__]