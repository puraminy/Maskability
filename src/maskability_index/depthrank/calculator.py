"""Public DepthRank calculator for seq2seq language models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from maskability_index.depthrank.interfaces import DepthRankResult, DepthRankTokenization
from maskability_index.depthrank.ranking import compute_relation_mi, maskability_index
from maskability_index.depthrank.scoring import depthrank_from_ranks, rank_token


@dataclass(slots=True)
class DepthRankCalculator:
    """Compute DepthRank exactly as defined by the manuscript equations."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str | torch.device | None = None
    zero_based: bool = True

    def __post_init__(self) -> None:
        """Move the model to the configured evaluation device."""
        if self.device is None:
            self.device = next(self.model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, prompt: str, target: str) -> DepthRankTokenization:
        """Create the code representation of S = {h_1:m, r_1:n, t_1:k}."""
        prompt_ids = tuple(
            self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors=None)
        )
        target_ids = tuple(
            self.tokenizer.encode(target, add_special_tokens=False, return_tensors=None)
        )
        if not target_ids:
            raise ValueError("DepthRank requires a non-empty gold target tail.")
        return DepthRankTokenization(
            prompt_token_ids=prompt_ids,
            target_token_ids=target_ids,
            target_tokens=tuple(self.tokenizer.convert_ids_to_tokens(target_ids)),
        )

    @torch.no_grad()
    def compute_token_ranks(
        self, prompt: str, target: str
    ) -> tuple[DepthRankTokenization, tuple[int, ...]]:
        """Compute every Index(t_i | h_1:m, r_1:n, t_<i) for a target tail."""
        tokenization = self.tokenize(prompt, target)
        encoded_inputs = self.tokenizer(prompt, return_tensors="pt")
        labels = torch.tensor([tokenization.target_token_ids], dtype=torch.long)
        encoded_inputs = {key: value.to(self.device) for key, value in encoded_inputs.items()}
        labels = labels.to(self.device)
        outputs = self.model(**encoded_inputs, labels=labels)
        logits = outputs.logits[0, : labels.shape[1], :].detach().cpu().numpy()
        ranks = tuple(
            rank_token(logits[position], token_id, zero_based=self.zero_based)
            for position, token_id in enumerate(tokenization.target_token_ids)
        )
        return tokenization, ranks

    def compute(self, prompt: str, target: str) -> DepthRankResult:
        """Public API: compute DepthRank(S) for one rendered prompt and gold tail."""
        tokenization, token_ranks = self.compute_token_ranks(prompt, target)
        return DepthRankResult(
            prompt=prompt,
            target=target,
            token_ids=tokenization.target_token_ids,
            tokens=tokenization.target_tokens,
            token_ranks=token_ranks,
            depthrank=depthrank_from_ranks(token_ranks),
        )

    def compute_many(self, examples: Iterable[tuple[str, str]]) -> list[DepthRankResult]:
        """Compute DepthRank for many `(prompt, target)` examples."""
        return [self.compute(prompt, target) for prompt, target in examples]

    def compute_maskability_index(
        self, prompting: Sequence[float], masked_prompting: Sequence[float]
    ) -> float:
        """Compute MI from two samples of DepthRank values."""
        if not prompting or not masked_prompting:
            raise ValueError("MI requires non-empty prompting and masked prompting samples.")
        dr_prompting = float(sum(prompting) / len(prompting))
        dr_masked = float(sum(masked_prompting) / len(masked_prompting))
        return maskability_index(dr_prompting, dr_masked)

    def compute_relation_maskability_index(
        self,
        prompting: Mapping[str, Sequence[DepthRankResult]],
        masked_prompting: Mapping[str, Sequence[DepthRankResult]],
    ) -> dict[str, float]:
        """Compute MI for relations that have both prompting and masked-prompting results."""
        return compute_relation_mi(prompting, masked_prompting)
