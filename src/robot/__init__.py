"""Robot Core — Local-first humanoid robot brain."""

import warnings

# Suppress upstream sounddevice numpy 2.x data.shape deprecation warning
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="sounddevice",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*Setting the shape on a NumPy array.*",
)

__version__ = "1.0.0"

