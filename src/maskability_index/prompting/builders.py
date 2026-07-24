"""Prompt builders that consume typed relation instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from maskability_index.datasets.atomic import RelationInstance
from maskability_index.templates import TemplateRegistry, atomic2020_registry


class PromptBuilder(Protocol):
    """Protocol for extensible prompt-generation strategies."""

    def build(self, instance: RelationInstance) -> str:
        """Build a prompt for one relation instance."""


@dataclass(frozen=True, slots=True)
class PrefixPromptBuilder:
    """Build unmasked prefix prompts for tail generation."""

    registry: TemplateRegistry = field(default_factory=atomic2020_registry)

    def build(self, instance: RelationInstance) -> str:
        return self.registry.get(instance.relation).prefix(instance)


@dataclass(frozen=True, slots=True)
class MaskedPromptBuilder:
    """Build masked span prompts for tail reconstruction."""

    registry: TemplateRegistry = field(default_factory=atomic2020_registry)
    mask_token: str = "<extra_id_0>"

    def build(self, instance: RelationInstance) -> str:
        return self.registry.get(instance.relation).masked(instance, mask_token=self.mask_token)


@dataclass(frozen=True, slots=True)
class FewShotPromptBuilder:
    """Build deterministic few-shot prompts from explicit demonstrations."""

    demonstrations: tuple[RelationInstance, ...]
    registry: TemplateRegistry = field(default_factory=atomic2020_registry)
    base_builder: PromptBuilder | None = None
    separator: str = "\n\n"

    def build(self, instance: RelationInstance) -> str:
        base_builder = self.base_builder or PrefixPromptBuilder(self.registry)
        demo_text = [self.registry.get(d.relation).full_text(d) for d in self.demonstrations]
        demo_text.append(base_builder.build(instance))
        return self.separator.join(demo_text)


def builder_from_style(style: str, *, demonstrations: tuple[RelationInstance, ...] = ()) -> PromptBuilder:
    """Instantiate a prompt builder from Hydra-configured style names."""

    if style == "prefix":
        return PrefixPromptBuilder()
    if style == "masked":
        return MaskedPromptBuilder()
    if style in {"few-shot", "few_shot"}:
        return FewShotPromptBuilder(demonstrations=demonstrations)
    raise ValueError(f"Unsupported prompt style {style!r}. Expected prefix, masked, or few-shot.")
