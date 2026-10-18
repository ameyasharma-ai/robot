"""Model catalog and registry for 2B Vision Language Models (VLM) and <4B Uncensored LLMs.

Provides metadata, capability tracking (tool usage, vision support, uncensored weights),
GGUF binary validation, and local path resolution for multi-model serving.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ModelCategory(str, Enum):
    """Categorical classification of local inference models."""

    VLM_2B = "vlm_2b"
    VLM_UNCENSORED = "vlm_uncensored"
    UNCENSORED_UNDER_4B = "uncensored_under_4b"
    STANDARD_TEXT = "standard_text"
    COMMUNITY_HUGGINGFACE = "community_huggingface"


@dataclass
class ModelMetadata:
    """Metadata specification for local GGUF models and vision projectors."""

    id: str
    name: str
    category: ModelCategory
    parameters: str
    repo_id: str
    filename: str
    mmproj_filename: str | None
    is_vision: bool
    is_uncensored: bool
    supports_tools: bool
    recommended_context: int
    description: str

    def get_model_path(self, base_dir: Path | None = None) -> Path:
        """Return the target file path for the primary GGUF model.

        Args:
            base_dir: Optional base project root directory.

        Returns:
            Path: Resolved absolute or relative path to the model file.
        """
        root = base_dir or Path.cwd()
        return root / "models" / "llm" / self.filename

    def get_mmproj_path(self, base_dir: Path | None = None) -> Path | None:
        """Return the target file path for the multimodal vision projector (mmproj).

        Args:
            base_dir: Optional base project root directory.

        Returns:
            Path | None: Resolved path to the mmproj file if multimodal, else None.
        """
        if not self.mmproj_filename:
            return None
        root = base_dir or Path.cwd()
        return root / "models" / "vision" / self.mmproj_filename

    def is_downloaded(self, base_dir: Path | None = None) -> bool:
        """Check if both the model GGUF (and mmproj if vision) exist and have valid sizes.

        Args:
            base_dir: Optional base project root directory.

        Returns:
            bool: True if all required files exist on disk with non-trivial size.
        """
        m_path = self.get_model_path(base_dir)
        if not m_path.exists() or m_path.stat().st_size < 1_000_000:
            return False

        if self.is_vision and self.mmproj_filename:
            mm_path = self.get_mmproj_path(base_dir)
            if not mm_path or not mm_path.exists() or mm_path.stat().st_size < 1_000_000:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Model metadata serialized dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value if isinstance(self.category, ModelCategory) else str(self.category),
            "parameters": self.parameters,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "mmproj_filename": self.mmproj_filename,
            "is_vision": self.is_vision,
            "is_uncensored": self.is_uncensored,
            "supports_tools": self.supports_tools,
            "recommended_context": self.recommended_context,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Construct ModelMetadata from dictionary representation.

        Args:
            data: Serialized model metadata.

        Returns:
            ModelMetadata: Hydrated dataclass instance.
        """
        cat_raw = data.get("category", "community_huggingface")
        try:
            category = ModelCategory(cat_raw)
        except ValueError:
            category = ModelCategory.COMMUNITY_HUGGINGFACE

        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            category=category,
            parameters=str(data.get("parameters", "Unknown")),
            repo_id=str(data.get("repo_id", "")),
            filename=str(data["filename"]),
            mmproj_filename=data.get("mmproj_filename"),
            is_vision=bool(data.get("is_vision", False)),
            is_uncensored=bool(data.get("is_uncensored", False)),
            supports_tools=bool(data.get("supports_tools", True)),
            recommended_context=int(data.get("recommended_context", 16384)),
            description=str(data.get("description", "")),
        )


class ModelRegistry:
    """Central registry and resolver for 2B VLM and uncensored <4B models."""

    DEFAULT_VLM_MODEL_ID = "qwen2-vl-2b-instruct"

    CATALOG: dict[str, ModelMetadata] = {
        # ─── 2B Vision Language Models with Tool Calling ─────────────────────────
        "qwen2-vl-2b-instruct": ModelMetadata(
            id="qwen2-vl-2b-instruct",
            name="Qwen2-VL 2B Instruct",
            category=ModelCategory.VLM_2B,
            parameters="2B",
            repo_id="bartowski/Qwen2-VL-2B-Instruct-GGUF",
            filename="Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
            mmproj_filename="mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
            is_vision=True,
            is_uncensored=False,
            supports_tools=True,
            recommended_context=16384,
            description="Leading 2B Vision-Language Model with native ChatML tool calling and OCR.",
        ),
        "qwen2-vl-2b-instruct-abliterated": ModelMetadata(
            id="qwen2-vl-2b-instruct-abliterated",
            name="Qwen2-VL 2B Instruct Abliterated",
            category=ModelCategory.VLM_UNCENSORED,
            parameters="2B",
            repo_id="mradermacher/Qwen2-VL-2B-Instruct-abliterated-GGUF",
            filename="Qwen2-VL-2B-Instruct-abliterated.Q4_K_M.gguf",
            mmproj_filename="Qwen2-VL-2B-Instruct-abliterated.mmproj-fp16.gguf",
            is_vision=True,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=16384,
            description="Uncensored 2B Vision Language Model with complete refusal ablation and tool calling.",
        ),
        "smolvlm-instruct-2b": ModelMetadata(
            id="smolvlm-instruct-2b",
            name="SmolVLM 2.2B Instruct",
            category=ModelCategory.VLM_2B,
            parameters="2.2B",
            repo_id="ggml-org/SmolVLM-Instruct-GGUF",
            filename="SmolVLM-Instruct-Q4_K_M.gguf",
            mmproj_filename="mmproj-SmolVLM-Instruct-f16.gguf",
            is_vision=True,
            is_uncensored=False,
            supports_tools=False,
            recommended_context=8192,
            description="HuggingFace lightweight 2.2B visual reasoning model with low memory footprint.",
        ),
        "moondream2-vision": ModelMetadata(
            id="moondream2-vision",
            name="Moondream2 1.8B",
            category=ModelCategory.VLM_2B,
            parameters="1.8B",
            repo_id="moondream/moondream2-gguf",
            filename="moondream2-text-model-f16.gguf",
            mmproj_filename="moondream2-mmproj-f16.gguf",
            is_vision=True,
            is_uncensored=False,
            supports_tools=False,
            recommended_context=4096,
            description="Ultra-fast 1.8B visual feature query model for instant scene understanding.",
        ),

        # ─── Uncensored / Abliterated Models Under 4B with Tool Calling ──────────
        "qwen2.5-coder-3b-abliterated": ModelMetadata(
            id="qwen2.5-coder-3b-abliterated",
            name="Qwen2.5 Coder 3B Instruct Abliterated",
            category=ModelCategory.UNCENSORED_UNDER_4B,
            parameters="3B",
            repo_id="bartowski/Qwen2.5-Coder-3B-Instruct-abliterated-GGUF",
            filename="Qwen2.5-Coder-3B-Instruct-abliterated-Q4_0.gguf",
            mmproj_filename=None,
            is_vision=False,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=16384,
            description="Premier 3B coding and function-calling model with abliterated refusal vectors.",
        ),
        "huihui-qwen3.5-4b-abliterated": ModelMetadata(
            id="huihui-qwen3.5-4b-abliterated",
            name="Huihui Qwen 3.5 4B Abliterated",
            category=ModelCategory.UNCENSORED_UNDER_4B,
            parameters="4B",
            repo_id="mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF",
            filename="Huihui-Qwen3.5-4B-abliterated.Q4_K_M.gguf",
            mmproj_filename=None,
            is_vision=False,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=16384,
            description="Uncensored 4B drop-in companion model with 100% Qwen 3.5 architectural parity.",
        ),
        "llama-3.2-3b-uncensored": ModelMetadata(
            id="llama-3.2-3b-uncensored",
            name="Llama 3.2 3B Instruct Uncensored",
            category=ModelCategory.UNCENSORED_UNDER_4B,
            parameters="3B",
            repo_id="bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF",
            filename="Llama-3.2-3B-Instruct-uncensored-Q4_0.gguf",
            mmproj_filename=None,
            is_vision=False,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=8192,
            description="Llama 3.2 3B fine-tuned for high compliance and uninhibited creative interaction.",
        ),
        "hermes-3-llama-3.2-3b": ModelMetadata(
            id="hermes-3-llama-3.2-3b",
            name="Nous Hermes 3 Llama 3.2 3B",
            category=ModelCategory.UNCENSORED_UNDER_4B,
            parameters="3B",
            repo_id="bartowski/Hermes-3-Llama-3.2-3B-GGUF",
            filename="Hermes-3-Llama-3.2-3B-Q4_K_M.gguf",
            mmproj_filename=None,
            is_vision=False,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=16384,
            description="Nous Research flagship steerable agent model with structured tool/function calling.",
        ),
        "llama-3.2-1b-uncensored": ModelMetadata(
            id="llama-3.2-1b-uncensored",
            name="Llama 3.2 1B Instruct Uncensored",
            category=ModelCategory.UNCENSORED_UNDER_4B,
            parameters="1.2B",
            repo_id="mradermacher/Llama-3.2-1B-Instruct-Uncensored-GGUF",
            filename="Llama-3.2-1B-Instruct-Uncensored.Q4_K_M.gguf",
            mmproj_filename=None,
            is_vision=False,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=8192,
            description="Ultra-fast 1.2B parameter uncensored model for low-latency (<80ms) edge inference.",
        ),
        "smollm2-1.7b-abliterated": ModelMetadata(
            id="smollm2-1.7b-abliterated",
            name="SmolLM2 1.7B Instruct Abliterated",
            category=ModelCategory.UNCENSORED_UNDER_4B,
            parameters="1.7B",
            repo_id="mradermacher/SmolLM2-1.7B-Instruct-abliterated-GGUF",
            filename="SmolLM2-1.7B-Instruct-abliterated.Q4_K_M.gguf",
            mmproj_filename=None,
            is_vision=False,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=8192,
            description="Compact 1.7B reasoning model with refusal abliteration for edge computing.",
        ),
        "qwen2.5-0.5b-abliterated": ModelMetadata(
            id="qwen2.5-0.5b-abliterated",
            name="Qwen2.5 0.5B Instruct Abliterated",
            category=ModelCategory.UNCENSORED_UNDER_4B,
            parameters="0.5B",
            repo_id="mradermacher/Qwen2.5-0.5B-Instruct-abliterated-SFT-i1-GGUF",
            filename="Qwen2.5-0.5B-Instruct-abliterated-SFT.i1-Q4_K_M.gguf",
            mmproj_filename=None,
            is_vision=False,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=4096,
            description="Sub-500MB micro model with abliterated safety guardrails for resource starvation fallback.",
        ),
        "phi-3.5-mini-uncensored": ModelMetadata(
            id="phi-3.5-mini-uncensored",
            name="Phi 3.5 Mini 3.8B Uncensored",
            category=ModelCategory.UNCENSORED_UNDER_4B,
            parameters="3.8B",
            repo_id="bartowski/Phi-3.5-mini-instruct_Uncensored-GGUF",
            filename="Phi-3.5-mini-instruct_Uncensored-Q4_K_M.gguf",
            mmproj_filename=None,
            is_vision=False,
            is_uncensored=True,
            supports_tools=True,
            recommended_context=16384,
            description="Microsoft 3.8B high-density reasoning architecture with removed safety constraints.",
        ),
    }

    _initialized: bool = False
    _custom_models: dict[str, ModelMetadata] = {}

    @classmethod
    def ensure_initialized(cls, base_dir: Path | None = None) -> None:
        """Ensure custom models from disk and local files are loaded into CATALOG.

        Args:
            base_dir: Optional base project root directory.
        """
        if cls._initialized:
            return
        cls.load_custom_models(base_dir)
        cls.scan_and_register_local_files(base_dir)
        cls._initialized = True

    @classmethod
    def register_model(
        cls,
        metadata: ModelMetadata,
        persist: bool = True,
        base_dir: Path | None = None,
    ) -> None:
        """Register a new model dynamically into CATALOG and optionally persist to disk.

        Args:
            metadata: ModelMetadata instance to register.
            persist: Whether to save to models/llm/registry.json.
            base_dir: Optional base project root directory.
        """
        cls.CATALOG[metadata.id] = metadata
        cls._custom_models[metadata.id] = metadata

        if persist:
            cls._save_custom_registry(base_dir)

    @classmethod
    def unregister_model(
        cls,
        model_id: str,
        persist: bool = True,
        base_dir: Path | None = None,
    ) -> bool:
        """Remove a model from the custom catalog.

        Args:
            model_id: Identifier of model to unregister.
            persist: Whether to persist removal to disk.
            base_dir: Optional base project root directory.

        Returns:
            bool: True if model was removed, False otherwise.
        """
        key = model_id.lower().strip()
        removed = False
        if key in cls._custom_models:
            del cls._custom_models[key]
            removed = True
        if key in cls.CATALOG:
            del cls.CATALOG[key]
            removed = True

        if removed and persist:
            cls._save_custom_registry(base_dir)
        return removed

    @classmethod
    def load_custom_models(cls, base_dir: Path | None = None) -> list[ModelMetadata]:
        """Load persisted custom models from models/llm/registry.json into CATALOG.

        Args:
            base_dir: Optional base project root directory.

        Returns:
            list[ModelMetadata]: Loaded custom models.
        """
        root = base_dir or Path.cwd()
        reg_file = root / "models" / "llm" / "registry.json"
        if not reg_file.exists():
            return []

        loaded: list[ModelMetadata] = []
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                items = raw_data if isinstance(raw_data, list) else raw_data.get("models", [])
                for item in items:
                    try:
                        meta = ModelMetadata.from_dict(item)
                        cls.CATALOG[meta.id] = meta
                        cls._custom_models[meta.id] = meta
                        loaded.append(meta)
                    except Exception:
                        continue
        except Exception:
            pass
        return loaded

    @classmethod
    def _save_custom_registry(cls, base_dir: Path | None = None) -> None:
        """Save registered custom models to models/llm/registry.json.

        Args:
            base_dir: Optional base project root directory.
        """
        root = base_dir or Path.cwd()
        reg_dir = root / "models" / "llm"
        reg_dir.mkdir(parents=True, exist_ok=True)
        reg_file = reg_dir / "registry.json"

        data = [m.to_dict() for m in cls._custom_models.values()]
        tmp_file = reg_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp_file.replace(reg_file)
        except Exception:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    @classmethod
    def scan_and_register_local_files(cls, base_dir: Path | None = None) -> list[ModelMetadata]:
        """Scan models/llm/ directory for existing GGUF files and register any unregistered models.

        Args:
            base_dir: Optional base project root directory.

        Returns:
            list[ModelMetadata]: Newly discovered and registered models.
        """
        root = base_dir or Path.cwd()
        llm_dir = root / "models" / "llm"
        if not llm_dir.exists():
            return []

        # Map existing filenames in CATALOG
        existing_filenames = {m.filename for m in cls.CATALOG.values()}
        new_models: list[ModelMetadata] = []

        for p in llm_dir.glob("*.gguf"):
            if p.name in existing_filenames or p.stat().st_size < 1_000_000:
                continue

            if not cls.verify_gguf_header(p):
                continue

            stem = p.stem
            model_id = re.sub(r"[^a-zA-Z0-9]+", "_", stem).lower().strip("_")

            # Extract parameter hint (e.g. 1.5B, 3B, 7B, 0.5B, etc.)
            param_match = re.search(r"(\d+(?:\.\d+)?[bB])", stem)
            params = param_match.group(1).upper() if param_match else "Custom"

            is_vision = any(x in stem.lower() for x in ["vl", "vision", "moondream", "smolvlm"])
            is_uncensored = any(x in stem.lower() for x in ["uncensored", "abliterated", "sft"])

            # Formulate readable name
            clean_name = stem.replace("-", " ").replace("_", " ").title()

            meta = ModelMetadata(
                id=model_id,
                name=clean_name,
                category=ModelCategory.COMMUNITY_HUGGINGFACE,
                parameters=params,
                repo_id="local/discovered",
                filename=p.name,
                mmproj_filename=None,
                is_vision=is_vision,
                is_uncensored=is_uncensored,
                supports_tools=True,
                recommended_context=16384,
                description=f"Auto-discovered local GGUF model ({params}) for RTX 3050 CUDA.",
            )
            cls.register_model(meta, persist=True, base_dir=base_dir)
            existing_filenames.add(p.name)
            new_models.append(meta)

        return new_models

    @classmethod
    def get(cls, model_id: str) -> ModelMetadata | None:
        """Retrieve model metadata by identifier, normalized key, or filename.

        Args:
            model_id: Unique model identifier or filename.

        Returns:
            ModelMetadata | None: Matched model metadata or None if unknown.
        """
        cls.ensure_initialized()
        key = model_id.lower().strip()
        if key in cls.CATALOG:
            return cls.CATALOG[key]
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", key).strip("_")
        if normalized in cls.CATALOG:
            return cls.CATALOG[normalized]
        for m in cls.CATALOG.values():
            if m.filename.lower() in (key, f"{key}.gguf"):
                return m
        return None

    @classmethod
    def list_models(cls, category: ModelCategory | None = None) -> list[ModelMetadata]:
        """List all models optionally filtered by category.

        Args:
            category: Optional category filter.

        Returns:
            list[ModelMetadata]: Registered model specifications.
        """
        cls.ensure_initialized()
        if category is None:
            return list(cls.CATALOG.values())
        return [m for m in cls.CATALOG.values() if m.category == category]

    @classmethod
    def list_vlm_2b_models(cls) -> list[ModelMetadata]:
        """List all 2B vision language models.

        Returns:
            list[ModelMetadata]: Models supporting vision capabilities around 2B size.
        """
        cls.ensure_initialized()
        return [m for m in cls.CATALOG.values() if m.category == ModelCategory.VLM_2B or (m.is_vision and m.parameters in ("2B", "2.2B"))]

    @classmethod
    def list_uncensored_models(cls) -> list[ModelMetadata]:
        """List all uncensored and abliterated models under 4B parameters.

        Returns:
            list[ModelMetadata]: Models marked as uncensored.
        """
        cls.ensure_initialized()
        return [m for m in cls.CATALOG.values() if m.is_uncensored]

    @classmethod
    def get_default_model(cls) -> ModelMetadata:
        """Return the default recommended 2B vision model.

        Returns:
            ModelMetadata: Default model specification (Qwen2-VL 2B Instruct).
        """
        cls.ensure_initialized()
        return cls.CATALOG[cls.DEFAULT_VLM_MODEL_ID]

    @classmethod
    def verify_gguf_header(cls, filepath: Path) -> bool:
        """Verify the binary GGUF magic header (0x46554747 = 'GGUF') of a model file.

        Args:
            filepath: Path to the GGUF file to validate.

        Returns:
            bool: True if file exists and starts with valid GGUF magic bytes.
        """
        if not filepath.exists() or filepath.stat().st_size < 4:
            return False
        try:
            with open(filepath, "rb") as f:
                magic = f.read(4)
                return magic == b"GGUF"
        except Exception:
            return False

    @classmethod
    def resolve_paths(
        cls,
        model_id: str,
        base_dir: Path | None = None,
    ) -> tuple[Path, Path | None]:
        """Resolve absolute filesystem paths for a model's primary weights and mmproj.

        Args:
            model_id: Identifier of target model.
            base_dir: Base directory path (defaults to current working directory).

        Returns:
            tuple[Path, Path | None]: Tuple of (model_file_path, mmproj_file_path_or_none).

        Raises:
            KeyError: If the model_id is not found in the registry.
        """
        cls.ensure_initialized(base_dir)
        meta = cls.get(model_id)
        if meta is None:
            raise KeyError(f"Unknown model identifier: '{model_id}'")
        return meta.get_model_path(base_dir), meta.get_mmproj_path(base_dir)
