from importlib.metadata import PackageNotFoundError, version

from .autoRIFT import autoRIFT

_pip_name = 'geo_autoRIFT'
try:
    __version__ = version(_pip_name)
except PackageNotFoundError:
    print(f'autoRIFT package is not installed!\n'
          f'From the top of this repository, install in editable/develop mode via:\n'
          f'   python -m pip install -e .\n'
          f'Or, to just get the version number use:\n'
          f'   python setup.py --version')

__all__ = [
    '__version__',
    'autoRIFT',
]
