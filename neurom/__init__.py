from .main import main
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neurom")

    __all__ = [main,__version__]
except:
     __all__ = [main]