"""Robot Core — Local-first humanoid robot brain."""

import warnings
import numpy as _np

# Suppress upstream sounddevice numpy 2.x data.shape deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import sounddevice as _sd
    # Clean monkey-patch: sounddevice 0.4.x assigns data.shape = -1, channels on a NumPy 2.x buffer,
    # which raises a DeprecationWarning on every single audio block. Using .reshape() fixes the root cause.
    def _clean_sd_array(buffer, channels, dtype):
        return _np.frombuffer(buffer, dtype=dtype).reshape(-1, channels)

    _sd._array = _clean_sd_array
except Exception:
    pass

__version__ = "1.0.0"

