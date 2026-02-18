"""
PRISM: Progressive Refinement with Interpretable Sequential Modeling

A regression methodology that automatically discovers optimal non-linear
transformations while providing exact sequential variance decomposition.
"""

from .regressor import PRISMRegressor, fit_prism, apply_transform

__version__ = "0.1.0"
__author__ = "Gavin Symanowitz"

__all__ = ["PRISMRegressor", "fit_prism", "apply_transform"]
