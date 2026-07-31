# Pipeline Audit and Manuscript Discrepancy Report

## Manuscript source of truth

The manuscript defines the experiment as ATOMIC2020 knowledge-base completion over nine relations, with T5-base, AdaFactor at learning rate `0.0001`, batch size `8`, `3` epochs, MI at `n=5`, and DepthRank on a held-out set of `100` heads per relation with up to three reference tails.

## Discrepancies found and resolved

1. **Heldout split discipline**: the runner previously loaded only an evaluation split in the top-level run path and did not validate train/heldout separation before training. The pipeline now loads configured train and heldout splits in Phase 1 and raises a descriptive error on triple overlap.
2. **Few-shot invariant**: training examples were sampled inside each training call without a centralized invariant check. The pipeline now constructs the few-shot set once in Phase 2 and validates exactly `number_of_relations × n_shot` examples, with exactly `n_shot` examples per relation.
3. **Demonstration leakage risk**: few-shot demonstrations were derived from the evaluation loader whenever few-shot mode was enabled. The implementation now only uses demonstrations when explicitly enabled and draws them from the training split, not heldout.
4. **Missing relation handling**: missing configured relations were previously logged as warnings. Because every configured relation is scientifically meaningful in the manuscript, the runner now raises an error if a selected relation is absent from a loaded split.
5. **Progress visibility**: several long phases had little structured progress reporting. The runner now logs numbered phases and DepthRank progress; dataset loading now uses the standard logging module instead of `print()`.
6. **Inference mode**: prediction generation used `torch.no_grad()`. It now uses `torch.inference_mode()` when PyTorch is available, preserving outputs while reducing inference overhead.

## Conservative choices

Optional predictions, plots, and LaTeX remain configurable outputs and are written after DepthRank/MI computation. They do not feed back into scientific metrics. Trainer-loop validation remains separate from heldout DepthRank/MI evaluation.
