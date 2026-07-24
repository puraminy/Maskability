# DepthRank Scientific Specification

This document extracts every mathematical definition related to DepthRank and Maskability Index from `paper/paper.tex` and maps each equation to implementation code.

## Source definitions

### Knowledge-base completion tuple

The paper defines each instance as a relational triple:

\[
(h, r, t)
\]

Variables and symbols:

- \(h\): head entity or event, represented as natural language text.
- \(r\): semantic relation label or its natural-language relation phrase.
- \(t\): gold tail sequence, represented as natural language text.

Interpretation: given \(h\) and \(r\), a model must generate a plausible tail \(t\).

Code mapping:

- `maskability_index.datasets.atomic.RelationInstance`
- `DepthRankCalculator.compute(...)` accepts `prompt` derived from \(h,r\) and `target` derived from \(t\).

## DepthRank definitions

### Tokenized triple

The manuscript defines a tokenized triple as:

\[
S = \{h_{1:m},\; r_{1:n},\; t_{1:k}\},
\]

Variables and symbols:

- \(S\): one tokenized triple under a chosen template family.
- \(h_{1:m}\): sequence of \(m\) tokens representing the head.
- \(r_{1:n}\): sequence of \(n\) tokens representing the relation or relation phrase.
- \(t_{1:k}\): sequence of \(k\) tokens representing the gold tail.
- \(m\): number of head tokens.
- \(n\): number of relation tokens.
- \(k\): number of target tail tokens.

Mathematical interpretation: the model conditions on the head, relation phrase, and previously decoded gold tail tokens when ranking each next gold tail token.

Code mapping:

- `interfaces.DepthRankTokenization` records `prompt_token_ids` and `target_token_ids`.
- `DepthRankCalculator.tokenize(...)` produces the future-code representation of \(S\), with \(h_{1:m}, r_{1:n}\) already rendered into the prompt by the prompt/template layer.

### Per-token rank/index

For each tail token \(t_i\), the manuscript defines:

\[
\mathrm{Index}\bigl(t_i \mid h_{1:m}, r_{1:n}, t_{<i}\bigr).
\]

Variables and symbols:

- \(t_i\): the \(i\)-th gold tail token.
- \(t_{<i}\): all gold tail tokens preceding \(t_i\).
- \(\mathrm{Index}(\cdot)\): rank/index of the correct token in the model probability list sorted by descending probability.

Mathematical interpretation: run teacher-forced scoring. At each tail position, sort the model vocabulary by predicted probability for the next token and record the sorted-list position of the gold token.

Code mapping:

- `scoring.rank_token(...)` implements \(\mathrm{Index}\).
- `DepthRankCalculator.compute_token_ranks(...)` computes all token-level indices for \(t_{1:k}\).

### Full-tail DepthRank

The manuscript defines DepthRank as:

\[
\mathrm{DepthRank}(S) = \frac{1}{k}\sum_{i=1}^{k}\mathrm{Index}\bigl(t_i \mid h_{1:m}, r_{1:n}, t_{<i}\bigr).
\]

Variables and symbols:

- \(\mathrm{DepthRank}(S)\): arithmetic mean of the gold-tail token indices.
- \(k\): number of gold tail tokens.
- Other variables are as defined above.

Mathematical interpretation: lower values mean the gold tail tokens occur nearer the top of the model's sorted probability lists.

Code mapping:

- `scoring.depthrank_from_ranks(...)` implements the averaging equation.
- `DepthRankCalculator.compute(...)` returns the complete result for one prompt/target pair.

## Relation-level mean DepthRank

The manuscript states that for relation \(r\) and few-shot sample size \(n\), mean DepthRank is computed across an \(n\)-sample under each template family:

\[
DR_{\mathrm{Prompting}}(r,n), \qquad DR_{\mathrm{MaskedPrompting}}(r,n).
\]

Variables and symbols:

- \(DR_{\mathrm{Prompting}}(r,n)\): mean DepthRank for relation \(r\), sample size \(n\), using prefix-style prompting.
- \(DR_{\mathrm{MaskedPrompting}}(r,n)\): mean DepthRank for relation \(r\), sample size \(n\), using masked-style prompting.
- \(r\): relation.
- \(n\): chosen few-shot sample size or evaluation sample count for the relation/template setting.

Mathematical interpretation: aggregate instance-level DepthRank values by arithmetic mean for each relation and template family.

Code mapping:

- `ranking.mean_depthrank(...)` computes the arithmetic mean.
- `ranking.relation_depthrank(...)` computes relation-level mean DepthRank from per-instance results.

## Maskability Index

The manuscript defines MI as:

\[
\mathrm{MI}(r,n) \;=\; \frac{DR_{\mathrm{Prompting}}(r,n) \;-\; DR_{\mathrm{MaskedPrompting}}(r,n)}{DR_{\mathrm{Prompting}}(r,n)}.
\]

Variables and symbols:

- \(\mathrm{MI}(r,n)\): relative DepthRank improvement of masked prompting over prefix prompting.
- \(DR_{\mathrm{Prompting}}(r,n)\): prefix-prompting mean DepthRank.
- \(DR_{\mathrm{MaskedPrompting}}(r,n)\): masked-prompting mean DepthRank.

Mathematical interpretation:

- \(\mathrm{MI}(r,n)>0\): masked prompting ranks gold tokens relatively higher.
- \(\mathrm{MI}(r,n)<0\): prefix prompting is relatively better.
- Magnitude gives relative advantage strength.

Code mapping:

- `ranking.maskability_index(...)` implements the MI equation.
- `DepthRankCalculator.compute_maskability_index(...)` exposes the public API for MI from two collections of DepthRank results.

## Algorithm description and computational procedure

Required inputs for one DepthRank computation:

1. A seq2seq model that returns next-token logits under teacher forcing.
2. A tokenizer for encoding prompt text and target tail text.
3. Prompt text containing \(h\) and \(r\) rendered by either prefix or masked templates.
4. Gold target tail text \(t\).

Procedure:

1. Tokenize prompt text into conditioning tokens corresponding to rendered \(h_{1:m}, r_{1:n}\).
2. Tokenize gold tail text into \(t_{1:k}\).
3. Run the model with the prompt and gold labels so each position predicts \(t_i\) conditioned on the prompt and \(t_{<i}\).
4. For every \(t_i\), sort vocabulary logits/probabilities in descending order and compute `Index(t_i | h, r, t_<i)`.
5. Compute `DepthRank(S)` as the arithmetic mean of the token indices.
6. Aggregate DepthRank values by relation/template with arithmetic means.
7. Compute MI from the two relation-level means.

Outputs:

- Per-token sorted-list indices.
- Per-instance DepthRank.
- Relation-level mean DepthRank.
- Maskability Index when both template-family means are supplied.

## Assumptions supported by the manuscript

- DepthRank is computed over gold target tokens, not generated predictions.
- The model is evaluated under teacher-forced conditioning on previous gold target tokens \(t_{<i}\).
- Prompting and MaskedPrompting are separate template families.
- Relation-level DepthRank values are arithmetic means over the selected sample.
- MI uses no normalization other than division by `DR_Prompting` as shown in the equation.

## Ambiguities recorded

Ambiguities that affect scientific reproducibility are recorded in `docs/AMBIGUITIES.md`.
