"""Tests for dataset normalization, templates, and prompt builders."""

from maskability_index.datasets import RelationInstance, iter_relation_instances
from maskability_index.prompting import FewShotPromptBuilder, MaskedPromptBuilder, PrefixPromptBuilder
from maskability_index.templates import CANONICAL_ATOMIC_PHRASES, atomic2020_registry


def test_every_atomic_relation_has_template() -> None:
    registry = atomic2020_registry()
    assert set(registry.list_relations()) == set(CANONICAL_ATOMIC_PHRASES)


def test_every_template_produces_valid_text() -> None:
    registry = atomic2020_registry()
    for relation in registry.list_relations():
        instance = RelationInstance("PersonX acts", relation, "a result", "train", f"id-{relation}")
        assert registry.get(relation).prefix(instance)
        assert "<extra_id_0>" in registry.get(relation).masked(instance)
        assert "a result" in registry.get(relation).full_text(instance)


def test_prompt_generation_is_deterministic() -> None:
    instance = RelationInstance("PersonX cooks", "xNeed", "buy food", "train", "1")
    demo = RelationInstance("Book", "AtLocation", "library", "train", "2")
    builders = [PrefixPromptBuilder(), MaskedPromptBuilder(), FewShotPromptBuilder((demo,))]
    for builder in builders:
        assert builder.build(instance) == builder.build(instance)


def test_iter_relation_instances_preserves_original_relation_names() -> None:
    rows = [{"id": "r1", "event": "PersonX eats", "xIntent": ["to be full"]}]
    instances = list(iter_relation_instances(rows, split="dev"))
    assert instances == [RelationInstance("PersonX eats", "xIntent", "to be full", "dev", "r1:xIntent:0")]
