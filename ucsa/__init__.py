"""UCSA top-level package.

Unified Cognitive State Architecture (UCSA) is a research-grade foundation model
whose computation revolves around a single persistent differentiable cognitive
state. Every operator acts on this state; every output is a projection of it.

See :mod:`ucsa.models.ucsa` for the top-level model definition.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "__author__",
]

__version__: str = "0.1.0"
__author__: str = "sachin"