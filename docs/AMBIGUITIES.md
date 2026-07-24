# Ambiguities in the DepthRank Manuscript Specification

The implementation documents, but does not silently hide, the following ambiguities in the paper.

1. **Rank base / index origin.** The paper writes `Index(...)` and states that a model prediction has rank zero by definition. This implies zero-based sorted-list indexing, so the implementation defaults to zero-based ranks. Some ranking literature uses one-based ranks.
2. **Tie handling.** The paper does not define how equal probabilities/logits are ranked. The implementation uses the stable sorted index returned by descending sort; exact ties therefore follow backend ordering.
3. **Special tokens in target length `k`.** The paper defines `t_{1:k}` as tail tokens but does not specify whether tokenizer-added special tokens such as EOS are included. The implementation defaults to tokenizer output with `add_special_tokens=False` for target text so `k` covers only the lexical target tokens.
4. **MaskedPrompting target format for T5.** The paper says masked prompting inserts unique mask token(s) and expects recovery of the masked span, but does not specify whether the decoder target includes sentinel tokens such as `<extra_id_0> tail <extra_id_1>`. The implementation scores the gold tail text itself, matching the DepthRank equation over `t_{1:k}`.
5. **Multiple reference tails per head.** The paper says held-out heads may have up to three reference tails, but the DepthRank equation is defined for a single tail sequence. The implementation treats each `(h,r,t)` triple as one DepthRank instance and averages supplied instances.
6. **MI at zero prefix DepthRank.** The paper defines MI with `DR_Prompting(r,n)` in the denominator but does not specify behavior when that value is zero. The implementation raises `ZeroDivisionError` because the equation is mathematically undefined.
