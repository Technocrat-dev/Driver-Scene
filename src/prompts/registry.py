"""Prompt registry for managing and iterating over prompt variants."""

from __future__ import annotations

from dataclasses import dataclass

from src.prompts.templates import PROMPT_TEMPLATES


@dataclass
class PromptVariant:
    """A registered prompt variant with metadata."""

    id: str
    strategy: str
    description: str
    template: str


class PromptRegistry:
    """Registry for managing prompt variants."""

    def __init__(self):
        self._variants: dict[str, PromptVariant] = {}
        self._load_builtin()

    def _load_builtin(self) -> None:
        """Load all built-in prompt templates."""
        for variant_id, info in PROMPT_TEMPLATES.items():
            self._variants[variant_id] = PromptVariant(
                id=variant_id,
                strategy=info["strategy"],
                description=info["description"],
                template=info["template"],
            )

    def get(self, variant_id: str) -> PromptVariant:
        """Get a prompt variant by ID."""
        if variant_id not in self._variants:
            available = ", ".join(sorted(self._variants.keys()))
            raise KeyError(f"Prompt '{variant_id}' not found. Available: {available}")
        return self._variants[variant_id]

    def list_all(self) -> list[PromptVariant]:
        """List all registered prompt variants."""
        return list(self._variants.values())

    def list_ids(self) -> list[str]:
        """List all variant IDs."""
        return list(self._variants.keys())

    def register(
        self,
        variant_id: str,
        strategy: str,
        description: str,
        template: str,
    ) -> None:
        """Register a custom prompt variant."""
        self._variants[variant_id] = PromptVariant(
            id=variant_id,
            strategy=strategy,
            description=description,
            template=template,
        )

    def __len__(self) -> int:
        return len(self._variants)

    def __iter__(self):
        return iter(self._variants.values())


# Global registry instance
registry = PromptRegistry()
