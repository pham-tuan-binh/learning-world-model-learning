"""
Inverse Dynamics Package.

Provides the Latent Action Model for learning actions from video:
- LatentActionModel: Complete encoder-quantizer-decoder model
- LatentActionsEncoder: Infers actions from frame pairs
- LatentActionsDecoder: Predicts next frame from current frame + action
"""

from .models import (
    LatentActionModel,
    LatentActionsEncoder,
    LatentActionsDecoder,
)

__all__ = [
    "LatentActionModel",
    "LatentActionsEncoder",
    "LatentActionsDecoder",
]
