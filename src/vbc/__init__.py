"""Vector-Based Comparison (VBC) for clinical endpoints."""

from .core import VBCResult, cosine_angle, decompose, transform_endpoints

__all__ = [
    "VBCResult",
    "cosine_angle",
    "decompose",
    "transform_endpoints",
]

__version__ = "0.1.0"
