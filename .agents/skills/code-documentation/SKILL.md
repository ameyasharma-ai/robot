---
name: code-documentation
description: >-
  Enforce comprehensive, crystal-clear documentation, type annotations, and docstrings across all code written or modified so it remains easily understandable, maintainable, and self-documenting.
---

# Code Documentation Standards Skill

Use this skill whenever you write new code, refactor existing functions, add classes, build API endpoints, or modify frontend scripts in this codebase.

The goal is to ensure that all code is **self-documenting, crystal clear, and easy to understand** for both human developers and AI pair programmers.

---

## 1. Python Documentation Standards

### 1.1 Function & Method Docstrings (Google / PEP 257 Format)
Every public function, coroutine, and class method must include a comprehensive docstring structured as follows:

```python
async def stream_audio_chunks(
    audio_stream: AsyncIterator[bytes],
    sample_rate: int = 16000,
    chunk_duration_ms: int = 50,
) -> AsyncIterator[np.ndarray]:
    """Slice an asynchronous raw byte stream into normalized float32 PCM numpy chunks.

    Buffers incoming raw PCM16 audio bytes until the duration threshold is reached,
    applies float32 normalization in range [-1.0, 1.0], and yields numpy arrays
    suitable for immediate VAD or streaming STT consumption.

    Args:
        audio_stream: Asynchronous generator yielding raw 16-bit PCM byte slices.
        sample_rate: Audio sampling frequency in Hertz (default: 16000 Hz).
        chunk_duration_ms: Target chunk slice window in milliseconds (default: 50 ms).

    Yields:
        1D numpy.ndarray of shape (N,) with float32 values normalized between -1.0 and 1.0.

    Raises:
        ValueError: If sample_rate is non-positive or chunk_duration_ms is less than 10.
        AudioStreamExhausted: If the underlying audio stream closes prematurely.
    """
```

#### Mandatory Docstring Sections:
- **One-line summary**: Imperative mood ("Slice an asynchronous raw byte stream...", NOT "Slices" or "This function slices").
- **Extended description**: Explains non-obvious logic, algorithmic strategy, thread-safety, or async lifecycle.
- **Args**: Explains every argument, its purpose, expected units (e.g. `ms`, `Hz`, `dB`), and default behavior.
- **Returns / Yields**: Concrete return type, shape (if array/tensor), and structural dictionary keys.
- **Raises**: Explicitly document known exceptions raised or handled.

---

### 1.2 Class Docstrings
Classes must detail their architectural role, thread-safety, hardware binding, and lifecycle:

```python
class StreamingSTT:
    """Real-time streaming speech-to-text engine using parallel chunked transcription.

    Maintains a rolling audio ring buffer, detects acoustic pauses via energy-dip
    thresholding, and offloads transcription chunks to background worker tasks on
    the AMD Radeon 680M iGPU to achieve sub-300ms transcription latency.

    Attributes:
        model: Whisper/Moonshine inference session loaded in memory.
        sample_rate: Audio sample frequency (default 16000 Hz).
        energy_threshold: RMS amplitude threshold for silence/voice demarcation.
        is_running: Boolean flag indicating if background worker loop is active.
    """
```

---

### 1.3 Strict Type Annotations
Every function signature must be 100% typed using modern Python 3.12+ type hints:
- Use `str | None` instead of `Optional[str]`.
- Use `list[T]`, `dict[K, V]`, `set[T]`, `tuple[T, ...]` instead of legacy `typing.List`, etc.
- Use `Callable[[ArgType], ReturnType]` for callback functions.
- Use `Path` from `pathlib` for all filesystem paths.

---

### 1.4 Architecture & Hardware Context Comments
Include explanatory inline comments for non-obvious engineering decisions:
- **Hardware Allocations**: Explain why code targets NVIDIA RTX 3050 (4GB VRAM) vs. AMD Radeon 680M vs. CPU AVX2.
- **Latency Budgets**: Mention timing constraints (e.g. `# Sub-10ms holding speech to prevent dead air`).
- **Audio / Device Caveats**: Explain quirks like ALSA device contention, PipeWire sink latency, or PortAudio ring buffers.

---

## 2. Frontend Documentation Standards (JavaScript & CSS)

### 2.1 JSDoc on JavaScript Functions & Classes
In `app.js`, `AnimationSystem.js`, `CharacterEngine.js`, all functions must have JSDoc blocks:

```javascript
/**
 * Triggers a 3D avatar gesture animation and updates the HUD status badge.
 *
 * @param {string} gestureName - Identifier of gesture ('nod', 'shrug', 'heart_gesture').
 * @param {number} [duration=1.5] - Animation duration in seconds.
 * @returns {Promise<boolean>} Resolves to true when the gesture playback completes.
 */
async triggerGesture(gestureName, duration = 1.5) {
  // ...
}
```

### 2.2 CSS Sectional Banners & Token Explanations
Organize `index.css` with structured comment headers:

```css
/* ==========================================================================
   CYBERDECK RESPONSIVE LAYOUT & BREAKPOINTS
   - Desktop (>= 1024px): 3-column zero-scroll HUD
   - Tablet (768px - 1023px): 2-column adaptive grid
   - Mobile (< 768px): Stacked single-column with quick tab navigation
   ========================================================================== */
```

---

## 3. Preservation of Existing Documentation
- **Never delete existing docstrings**: When modifying or refactoring existing methods, keep or update the existing documentation.
- **Never truncate comments**: Retain architectural notes left by previous engineering sessions.

---

## 4. Verification Checklist
Before completing any task involving code changes:
1. Every new or modified function has a complete docstring with `Args:`, `Returns:`, and `Raises:`.
2. All parameters and returns have explicit type annotations.
3. Complex algorithmic logic has inline comments explaining the *why*, not just the *what*.
4. Run code integrity check:
   ```bash
   .venv/bin/python .agents/skills/zero-import-errors/scripts/verify_imports.py
   ```
