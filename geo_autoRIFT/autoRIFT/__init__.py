from importlib.metadata import PackageNotFoundError, version

from .autoRIFT import autoRIFT

_pip_name = 'geo_autoRIFT'
try:
    __version__ = version(_pip_name)
except PackageNotFoundError:
    print(
        'autoRIFT package is not installed!\n'
        'From the top of this repository, install in editable/develop mode via:\n'
        '   python -m pip install -e .\n'
        'Or, to just get the version number use:\n'
        '   python setup.py --version'
    )

__all__ = [
    '__version__',
    'autoRIFT',
]
