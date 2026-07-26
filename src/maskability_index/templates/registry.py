"""Relation template registry for ATOMIC-style prompt verbalizers.

This module intentionally falls back to a generic template when a relation is
not registered, to avoid crashing experiments when dataset relation names differ
or contain unexpected keys (for example: `output`). The fallback emits a
warning so dataset/registry mismatches are visible in logs.
"""

from __future__ import annotations

import warnings
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
    """Registry of relation names to reusable prompt templates.

    If a template lookup fails, a generic fallback RelationTemplate is returned
    and a warning is emitted. This prevents crashes when datasets contain
    unexpected relation column names (for example `output`) while making the
    mismatch visible to the user.
    """

    def __init__(self, templates: Mapping[str, RelationTemplate] | None = None) -> None:
        """Create a registry from optional initial templates."""
        self._templates: dict[str, RelationTemplate] = dict(templates or {})

    def register(self, template: RelationTemplate) -> None:
        """Register a template, replacing any previous template for that relation."""
        self._templates[template.relation] = template

    def get(self, relation: str) -> RelationTemplate:
        """Return the template for a relation, preserving exact relation keys.

        If the relation is not registered, return a fallback template that uses
        the relation name as a phrase and emit a warning so callers can detect
        and (optionally) log the mismatch.
        """
        try:
            return self._templates[relation]
        except KeyError:
            warnings.warn(
                f"No template registered for relation {relation!r}; using fallback template."
            )
            # Derive a readable phrase from the relation key as a best-effort fallback
            phrase = relation.replace("_", " ")
            return RelationTemplate(relation=relation, phrase=phrase)

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
