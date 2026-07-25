"""Relation template registry for ATOMIC-style prompt verbalizers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from maskability_index.datasets.atomic import RelationInstance


@dataclass(frozen=True, slots=True)
class RelationTemplate:
    """Canonical text template for one relation."""

    relation: str
    phrase: str

    def prefix(self, instance: RelationInstance) -> str:
        """Render a prefix-style prompt that leaves the tail to be generated."""
        return f"{instance.head} {self.phrase}".strip()

    def masked(self, instance: RelationInstance, mask_token: str = "<extra_id_0>") -> str:
        """Render a masked prompt with a single explicit span to recover."""
        return f"{instance.head} {self.phrase} {mask_token}".strip()

    def full_text(self, instance: RelationInstance) -> str:
        """Render the complete verbalized triple including the gold tail."""
        return f"{self.prefix(instance)} {instance.tail}".strip()


class TemplateRegistry:
    """Registry of relation names to reusable prompt templates."""

    def __init__(self, templates: Mapping[str, RelationTemplate] | None = None) -> None:
        """Create a registry from optional initial templates."""
        self._templates: dict[str, RelationTemplate] = dict(templates or {})

    def register(self, template: RelationTemplate) -> None:
        """Register a template, replacing any previous template for that relation."""
        self._templates[template.relation] = template

    def get(self, relation: str) -> RelationTemplate:
        """Return the template for a relation, preserving exact relation keys."""
        try:
            return self._templates[relation]
        except KeyError as exc:
            raise KeyError(f"No template registered for relation {relation!r}.") from exc

    def list_relations(self) -> list[str]:
        """List relations with registered templates in deterministic order."""
        return sorted(self._templates)

CANONICAL_ATOMIC_PHRASES: dict[str, str] = {
    "AtLocation": "located at",
    "ObjectUse": "is used for",
    "UsedFor": "is used for",
    "CapableOf": "is capable of",
    "HasProperty": "has the property of",
    "FilledBy": "is filled by",
    "isFilledBy": "is filled by",
    "xAttr": "is seen as",
    "xIntent": "because they intended",
    "xNeed": "before that they need",
    "xWant": "after that they want",
    "xEffect": "as a result, PersonX",
    "xReact": "as a result, PersonX feels",
    "oEffect": "as a result, others",
    "oReact": "as a result, others feel",
    "oWant": "as a result, others want",
    "Causes": "causes",
    "CausesDesire": "causes the desire for",
    "CreatedBy": "is created by",
    "DefinedAs": "is defined as",
    "Desires": "desires",
    "HasA": "has",
    "HasFirstSubevent": "begins with",
    "HasLastSubevent": "ends with",
    "HasPainCharacter": "has pain described as",
    "HasPainIntensity": "has pain intensity",
    "HasPrerequisite": "requires",
    "HasSubEvent": "includes the event",
    "HinderedBy": "can be hindered by",
    "InheritsFrom": "inherits from",
    "InstanceOf": "is an instance of",
    "isAfter": "happens after",
    "isBefore": "happens before",
    "MadeOf": "is made of",
    "MadeUpOf": "is made up of",
    "MotivatedByGoal": "is motivated by the goal",
    "NotCapableOf": "is not capable of",
    "NotDesires": "does not desire",
    "PartOf": "is part of",
    "ReceivesAction": "can receive the action",
    "xReason": "because PersonX wanted",
}


def atomic2020_registry() -> TemplateRegistry:
    """Build the canonical ATOMIC2020 template registry."""
    registry = TemplateRegistry()
    for relation, phrase in CANONICAL_ATOMIC_PHRASES.items():
        registry.register(RelationTemplate(relation=relation, phrase=phrase))
    return registry
