# Maskability Index Scientific Specification

This document extracts the mathematical definitions related to the Maskability Index (MI) from `paper/paper.tex` and maps them to implementation code. The manuscript is the authoritative specification.

## Source task definition

Each knowledge-base completion instance is a triple

\[
(h, r, t)
\]

where:

- \(h\): head entity or event, represented as natural language text.
- \(r\): semantic relation label or natural-language relation phrase.
- \(t\): gold tail sequence, represented as natural language text.

The model is given \(h\) and \(r\) and must generate a plausible tail \(t\).

Code mapping:

- `maskability_index.datasets.atomic.RelationInstance`
- Prompt text is produced outside MI by the prompt/template layer.

## DepthRank quantities consumed by MI

For relation \(r\) and few-shot size \(n\) and held-out evaluation size, DepthRank is computed under two template families:

\[
DR_{\mathrm{Prompting}}(r,n), \qquad DR_{\mathrm{MaskedPrompting}}(r,n).
\]

Variables:

- \(DR_{\mathrm{Prompting}}(r,n)\): arithmetic mean DepthRank across the \(n\)-sample for relation \(r\) using unmasked prefix templates.
- \(DR_{\mathrm{MaskedPrompting}}(r,n)\): arithmetic mean DepthRank across the \(n\)-sample for relation \(r\) using masked templates.
- \(r\): relation.
- \(n\): selected few-shot condition and held-out DepthRank evaluation size.

Template-family definitions:

- **Prompting (P)**: unmasked prefix templates that rely on conditional generation.
- **MaskedPrompting (MP)**: templates with explicit unique mask token(s) corresponding to the tail; the model/decoder recovers the masked span.

Aggregation assumption from the manuscript: relation-level DepthRank values are arithmetic means over the selected sample.

Code mapping:

- `maskability_index.maskability.aggregation.mean_depthrank(...)`
- `maskability_index.maskability.aggregation.relation_depthrank(...)`

## Equation (7): Maskability Index

The manuscript defines MI as the relative DepthRank improvement of masked prompting over prompting:

\[
\mathrm{MI}(r,n) = \frac{DR_{\mathrm{Prompting}}(r,n) - DR_{\mathrm{MaskedPrompting}}(r,n)}{DR_{\mathrm{Prompting}}(r,n)}.
\]

Variables:

- \(\mathrm{MI}(r,n)\): Maskability Index for relation \(r\) at few-shot size \(n\) and held-out evaluation size.
- \(DR_{\mathrm{Prompting}}(r,n)\): prefix-prompting mean DepthRank.
- \(DR_{\mathrm{MaskedPrompting}}(r,n)\): masked-prompting mean DepthRank.

Interpretation:

- \(\mathrm{MI}(r,n)>0\): masked prompting ranks gold tokens relatively higher; relation is mask-friendly.
- \(\mathrm{MI}(r,n)<0\): prefix prompting is relatively better; relation is mask-resistant.
- Magnitude indicates strength of the relative advantage.

Code mapping:

- `maskability_index.maskability.metrics.maskability_index(...)`
- `maskability_index.maskability.calculator.MaskabilityCalculator.compute(...)`

## Threshold grouping used in experiments

The paper reports a 30% threshold at \(n=5\):

\[
\begin{cases}
\text{Mask-Filling} & \text{if } \mathrm{MI}\ge 0.30,\\[3pt]
\text{Map-Phrasal} & \text{otherwise.}
\end{cases}
\]

This threshold is an experimental grouping rule, not part of the MI equation. It is configurable in experiment code and optional in the public API.

Code mapping:

- `maskability_index.maskability.metrics.classify_maskability(...)`

## Algorithm

Inputs:

1. Relation identifier \(r\).
2. Sample size \(n\).
3. A non-empty sample of DepthRank values computed from Prompting templates.
4. A non-empty sample of DepthRank values computed from MaskedPrompting templates.

Procedure:

1. Compute \(DR_{\mathrm{Prompting}}(r,n)\) as the arithmetic mean of prefix DepthRank values.
2. Compute \(DR_{\mathrm{MaskedPrompting}}(r,n)\) as the arithmetic mean of masked DepthRank values.
3. Compute Equation (7) exactly: `(DR_Prompting - DR_MaskedPrompting) / DR_Prompting`.
4. Optionally assign a group using the manuscript's experimental threshold rule.

Outputs:

- relation
- few-shot size \(n\) and held-out evaluation size
- \(DR_{\mathrm{Prompting}}(r,n)\)
- \(DR_{\mathrm{MaskedPrompting}}(r,n)\)
- \(\mathrm{MI}(r,n)\)
- optional group label

## Assumptions

- MI consumes DepthRank values and does not duplicate DepthRank computation.
- DepthRank remains an independent module.
- MI performs no normalization beyond the division by \(DR_{\mathrm{Prompting}}(r,n)\) shown in Equation (7).
- Samples supplied to MI already correspond to the same relation and sample-size setting.

## Ambiguities

The manuscript does not define MI when \(DR_{\mathrm{Prompting}}(r,n)=0\). The implementation raises `ZeroDivisionError` because Equation (7) is mathematically undefined in that case. This is recorded in `docs/AMBIGUITIES.md`.
