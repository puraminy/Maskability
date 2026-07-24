# Prompt Infrastructure

Milestone 2 keeps prompting independent from training, evaluation, DepthRank, and Maskability Index code.

## Template storage

Canonical relation verbalizers live in `src/maskability_index/templates/registry.py` as `RelationTemplate` objects managed by `TemplateRegistry`.

The registry exposes three operations:

- `register(template)` adds or replaces a relation template.
- `get(relation)` returns the template for an exact ATOMIC relation name.
- `list_relations()` returns registered relations in deterministic order.

Templates are not embedded in prompt builders or dataset loaders. The default `atomic2020_registry()` creates one canonical template per supported ATOMIC2020 relation.

## Prompt generation

Prompt builders live in `src/maskability_index/prompting/builders.py` and consume `RelationInstance` dataclasses from `src/maskability_index/datasets/atomic.py`.

Supported builders are:

- `PrefixPromptBuilder`: renders a head and relation phrase as an unmasked continuation prompt.
- `MaskedPromptBuilder`: renders the same relation phrase followed by the T5-style `<extra_id_0>` span mask.
- `FewShotPromptBuilder`: prepends deterministic demonstration triples before the query prompt.

Hydra selects the prompting style through `experiment.prompting.style` with supported values `prefix`, `masked`, and `few-shot`. Code should call `builder_from_style()` rather than hardcoding a builder choice.

## Adding templates

To add a new relation template, register a `RelationTemplate` with a `TemplateRegistry`:

```python
from maskability_index.templates import RelationTemplate, atomic2020_registry

registry = atomic2020_registry()
registry.register(RelationTemplate(relation="NewRelation", phrase="natural phrase"))
```

For a new canonical ATOMIC relation, add the phrase to `CANONICAL_ATOMIC_PHRASES` and extend the template validation tests.

## Adding prompting styles

Future experiments such as alternative templates, instruction prompts, or chain-of-thought prompts should add a new `PromptBuilder` class implementing `build(instance: RelationInstance) -> str`.

Existing builders should not be modified unless their own behavior changes. Configuration dispatch can be extended in `builder_from_style()` or by injecting the new builder directly from experiment code.
