"""ADN: Adaptive Image Downscaling Network.

Unofficial, conference-quality reproduction of
"ADN: Adaptive Image Downscaling Network" (Pise & Ghosh).
"""

__version__ = "1.0.0"

from adn.models.adn import ADN, build_adn

__all__ = ["ADN", "build_adn", "__version__"]
