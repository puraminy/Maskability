# Scientific Ambiguities

This file records methodology details that could not be resolved from the manuscript, `docs/IMPLEMENTATION_SPEC.md`, and `docs/SOFTWARE_SPEC.md` without inventing behavior.

## Fine-tuning data size versus DepthRank evaluation size

The manuscript states that sample size was varied from 3 to 100 in Figure 1 and that MI is computed at `n=5`, while DepthRank is computed on a held-out set of 100 heads per relation with up to three reference tails. It does not explicitly state whether `n` is the number of fine-tuning examples per relation, the number of prompts used directly in MI aggregation, or both. The implementation treats `n=5` as the few-shot fine-tuning sample size and evaluates DepthRank on the independently configured held-out 100 heads.

## Template-family-specific fine-tuning

The manuscript reports Prompting and MaskedPrompting as separate template families and states that T5-base is fine-tuned on downstream reasoning templates, but it does not explicitly say whether one model is fine-tuned separately for each template family or one mixed-template model is fine-tuned once and evaluated under both prompts. The implementation fine-tunes one model per template family because that is the scientifically conservative way to compare each family under its own training/evaluation template without mixing objectives.

## Development split usage

The manuscript gives train/fine-tuning hyperparameters and a held-out DepthRank evaluation set, but does not name the validation/development split used during fine-tuning. The implementation uses the configured `dev_split` for trainer evaluation and `evaluation_split` for held-out DepthRank.

## Generation-quality evaluation details

The manuscript reports ROUGE and BERTScore tables, but it does not specify all generation decoding parameters, BERTScore model/language settings, or the exact aggregation procedure used to produce those tables. The implementation can export predictions with configured generation length and beam size, but ROUGE/BERTScore reproduction remains under-specified.

## Random sampling protocol

The manuscript lists deterministic hyperparameters and evaluation sizes, but does not specify random seeds or whether relation examples/heads are selected deterministically or randomly. The implementation keeps seeds and sampling strategies in YAML and defaults the paper reproduction to deterministic selection with seed 13.

## Rank base / index origin

The paper writes `Index(...)` and states that a model prediction has rank zero by definition. This implies zero-based sorted-list indexing, so the implementation defaults to zero-based ranks. Some ranking literature uses one-based ranks.

## Tie handling

The paper does not define how equal probabilities/logits are ranked. The implementation uses the stable sorted index returned by descending sort; exact ties therefore follow backend ordering.

## Special tokens in target length `k`

The paper defines `t_{1:k}` as tail tokens but does not specify whether tokenizer-added special tokens such as EOS are included. The implementation defaults to tokenizer output with `add_special_tokens=False` for target text so `k` covers only the lexical target tokens.

## MaskedPrompting target format for T5

The paper says masked prompting inserts unique mask token(s) and expects recovery of the masked span, but does not specify whether the decoder target includes sentinel tokens such as `<extra_id_0> tail <extra_id_1>`. The implementation scores the gold tail text itself, matching the DepthRank equation over `t_{1:k}`.

## Multiple reference tails per head

The paper says held-out heads may have up to three reference tails, but the DepthRank equation is defined for a single tail sequence. The implementation treats each `(h,r,t)` triple as one DepthRank instance and averages supplied instances.

## MI at zero prefix DepthRank

The paper defines MI with `DR_Prompting(r,n)` in the denominator but does not specify behavior when that value is zero. The implementation raises `ZeroDivisionError` because the equation is mathematically undefined.
