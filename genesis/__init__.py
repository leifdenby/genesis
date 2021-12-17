import seaborn as sns

from . import _version

__version__ = _version.get_versions()["version"]

# use seaborn color palette by default
sns.set(color_codes=True, style="ticks")
