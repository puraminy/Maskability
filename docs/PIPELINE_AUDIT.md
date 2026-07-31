# Experiment Pipeline Audit

## Manuscript-derived required pipeline

1. Load ATOMIC2020 triples preserving head, relation, and tail.
2. Select the nine reported relations: AtLocation, ObjectUse, xAttr, CapableOf, HasProperty, FilledBy/isFilledBy, xIntent, xNeed, and xWant.
3. Render one canonical natural-language template per relation for both Prompting and MaskedPrompting.
4. Construct a few-shot setting with `n=5` for MI.
5. Fine-tune T5-base with AdaFactor, learning rate 0.0001, batch size 8, and 3 epochs.
6. Evaluate DepthRank on a held-out set of 100 heads per relation, with up to three reference tails.
7. Compute MI as `(DR_Prompting - DR_MaskedPrompting) / DR_Prompting` and group relations with threshold 0.30.
8. Export DepthRank rows, MI tables, statistics, plots, and generated predictions/tables needed to support reported results.

## Discrepancies found before this change

- The generic `ExperimentRunner` loaded a pretrained model and computed DepthRank directly even when `training.enabled: true`; fine-tuning hyperparameters in YAML were not used by the reproduction pipeline.
- The reproduction config evaluated the `train` split rather than a held-out split, conflicting with the manuscript's held-out DepthRank evaluation description.
- The reproduction config did not expose all scientific training/generation parameters in YAML (`optimizer`, scheduler, warmup, target/input lengths, generation length, beam size, trainer seed, checkpoint strategies).
- The local dataset backend raised before trying CSV files, preventing local split fixtures and CSV reproduction data from being used reliably.
- Prediction export was disabled in the reproduction config, even though prediction generation is an explicit experiment phase and is needed for downstream generation-quality tables.

## Corrections implemented

- `ExperimentRunner` now runs template-family-specific fine-tuning before DepthRank when `training.enabled` is true.
- Train/dev/evaluation split names are now separate configuration values; paper reproduction uses train, validation, and test respectively.
- Fine-tuning sample construction uses configured few-shot size and sampling seed; DepthRank evaluation uses a separate held-out head/reference-tail sampler.
- Checkpoints are saved below the configured checkpoint directory per template family.
- Prediction export now uses the trained model for each corresponding template family.
- The reproduction YAML now contains the paper's optimizer, learning rate, epochs, batch size, evaluation size, maximum reference tails, relation list, MI threshold, seed, and generation settings.
